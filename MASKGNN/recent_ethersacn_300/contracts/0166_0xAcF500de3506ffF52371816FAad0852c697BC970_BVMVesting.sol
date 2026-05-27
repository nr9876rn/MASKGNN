// ===== FILE: contracts_-_bvm/utilities/BVMVesting.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

interface IERC20Min { function transfer(address, uint256) external returns (bool); function transferFrom(address, address, uint256) external returns (bool); function balanceOf(address) external view returns (uint256); }

contract BVMVesting is BVMFeeBase {
    uint256 public setupFeeBvm = 210 ether;

    struct Schedule {
        address creator;
        address beneficiary;
        address token;
        uint128 totalAmount;
        uint128 released;
        uint64  startAt;
        uint64  cliffAt;
        uint64  endAt;
        bool    revocable;
        bool    revoked;
    }
    Schedule[] public schedules;
    mapping(address => uint256[]) public schedulesOf;

    event ScheduleCreated(uint256 indexed id, address indexed creator, address indexed beneficiary, address token, uint128 total, uint64 startAt, uint64 cliffAt, uint64 endAt, bool revocable);
    event Released(uint256 indexed id, address indexed beneficiary, uint128 amount);
    event Revoked(uint256 indexed id, address indexed creator, uint128 refunded);
    event FeeChanged(uint256 prev, uint256 next);

    error BadParams();
    error NotBeneficiary();
    error NotCreator();
    error NothingDue();
    error NotRevocable();
    error AlreadyRevoked();
    error TransferFailed();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function create(
        address beneficiary, address token, uint128 totalAmount,
        uint64 startAt, uint64 cliffAt, uint64 endAt, bool revocable
    ) external returns (uint256 id) {
        if (beneficiary == address(0) || token == address(0)) revert BadParams();
        if (totalAmount == 0) revert BadParams();
        if (endAt <= startAt || cliffAt < startAt || cliffAt > endAt) revert BadParams();
        _payFee(setupFeeBvm);
        if (!IERC20Min(token).transferFrom(msg.sender, address(this), totalAmount)) revert TransferFailed();
        id = schedules.length;
        schedules.push(Schedule({
            creator: msg.sender, beneficiary: beneficiary, token: token,
            totalAmount: totalAmount, released: 0,
            startAt: startAt, cliffAt: cliffAt, endAt: endAt,
            revocable: revocable, revoked: false
        }));
        schedulesOf[beneficiary].push(id);
        emit ScheduleCreated(id, msg.sender, beneficiary, token, totalAmount, startAt, cliffAt, endAt, revocable);
    }

    function vested(uint256 id) public view returns (uint128) {
        Schedule storage s = schedules[id];
        if (s.revoked || block.timestamp < s.cliffAt) return 0;
        if (block.timestamp >= s.endAt) return s.totalAmount;
        uint256 elapsed = block.timestamp - uint256(s.startAt);
        uint256 total   = uint256(s.endAt) - uint256(s.startAt);
        return uint128((uint256(s.totalAmount) * elapsed) / total);
    }

    function releasable(uint256 id) public view returns (uint128) {
        Schedule storage s = schedules[id];
        uint128 v = vested(id);
        if (v <= s.released) return 0;
        return v - s.released;
    }

    function release(uint256 id) external {
        Schedule storage s = schedules[id];
        if (msg.sender != s.beneficiary) revert NotBeneficiary();
        uint128 amount = releasable(id);
        if (amount == 0) revert NothingDue();
        s.released += amount;
        if (!IERC20Min(s.token).transfer(s.beneficiary, amount)) revert TransferFailed();
        emit Released(id, s.beneficiary, amount);
    }

    function revoke(uint256 id) external {
        Schedule storage s = schedules[id];
        if (msg.sender != s.creator) revert NotCreator();
        if (!s.revocable) revert NotRevocable();
        if (s.revoked) revert AlreadyRevoked();
        uint128 v = vested(id);
        uint128 stillVested = v - s.released;
        if (stillVested > 0) {
            s.released += stillVested;
            if (!IERC20Min(s.token).transfer(s.beneficiary, stillVested)) revert TransferFailed();
            emit Released(id, s.beneficiary, stillVested);
        }
        uint128 refund = s.totalAmount - s.released;
        s.revoked = true;
        if (refund > 0) {
            if (!IERC20Min(s.token).transfer(s.creator, refund)) revert TransferFailed();
        }
        emit Revoked(id, s.creator, refund);
    }

    function totalSchedules() external view returns (uint256) { return schedules.length; }

    function setFee(uint256 next) external onlyOwner {
        emit FeeChanged(setupFeeBvm, next);
        setupFeeBvm = next;
    }
}


// ===== FILE: contracts_-_bvm/utilities/BVMFeeBase.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

interface IBVMTreasury {
    function collect(address payer, uint256 amount) external;
}

abstract contract BVMFeeBase {
    IBVMTreasury public immutable bvmTreasury;
    address      public owner;

    event FeeRouted(address indexed payer, uint256 amount, bytes4 indexed selector);
    event OwnershipTransferred(address indexed prev, address indexed next);

    error NotOwner();
    error ZeroAddress();

    modifier onlyOwner() { if (msg.sender != owner) revert NotOwner(); _; }

    constructor(address _treasury, address _owner) {
        if (_treasury == address(0) || _owner == address(0)) revert ZeroAddress();
        bvmTreasury = IBVMTreasury(_treasury);
        owner       = _owner;
    }

    function _payFee(uint256 amount) internal {
        if (amount == 0) return;
        bvmTreasury.collect(msg.sender, amount);
        emit FeeRouted(msg.sender, amount, msg.sig);
    }

    function transferOwnership(address next) external onlyOwner {
        if (next == address(0)) revert ZeroAddress();
        emit OwnershipTransferred(owner, next);
        owner = next;
    }

    function renounceOwnership() external onlyOwner {
        emit OwnershipTransferred(owner, address(0));
        owner = address(0);
    }
}
