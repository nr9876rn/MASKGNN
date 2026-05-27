// ===== FILE: contracts_-_bvm/utilities/BVMLocker.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

interface IERC20Min { function transfer(address, uint256) external returns (bool); function transferFrom(address, address, uint256) external returns (bool); }

contract BVMLocker is BVMFeeBase {
    uint256 public lockFeeBvm = 210 ether;

    struct Lock {
        address creator;
        address beneficiary;
        address token;
        uint128 amount;
        uint64  unlockAt;
        bool    withdrawn;
    }
    Lock[] public locks;
    mapping(address => uint256[]) public locksOf;

    event Locked(uint256 indexed id, address indexed creator, address indexed beneficiary, address token, uint128 amount, uint64 unlockAt);
    event Unlocked(uint256 indexed id, address indexed beneficiary, uint128 amount);
    event FeeChanged(uint256 prev, uint256 next);

    error BadParams();
    error NotBeneficiary();
    error TooEarly();
    error AlreadyOut();
    error TransferFailed();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function lock(address beneficiary, address token, uint128 amount, uint64 unlockAt) external returns (uint256 id) {
        if (beneficiary == address(0) || token == address(0)) revert BadParams();
        if (amount == 0 || unlockAt <= block.timestamp) revert BadParams();
        _payFee(lockFeeBvm);
        if (!IERC20Min(token).transferFrom(msg.sender, address(this), amount)) revert TransferFailed();
        id = locks.length;
        locks.push(Lock({
            creator: msg.sender, beneficiary: beneficiary, token: token,
            amount: amount, unlockAt: unlockAt, withdrawn: false
        }));
        locksOf[beneficiary].push(id);
        emit Locked(id, msg.sender, beneficiary, token, amount, unlockAt);
    }

    function withdraw(uint256 id) external {
        Lock storage l = locks[id];
        if (l.withdrawn) revert AlreadyOut();
        if (msg.sender != l.beneficiary) revert NotBeneficiary();
        if (block.timestamp < l.unlockAt) revert TooEarly();
        l.withdrawn = true;
        if (!IERC20Min(l.token).transfer(l.beneficiary, l.amount)) revert TransferFailed();
        emit Unlocked(id, l.beneficiary, l.amount);
    }

    function totalLocks() external view returns (uint256) { return locks.length; }

    function setFee(uint256 next) external onlyOwner {
        emit FeeChanged(lockFeeBvm, next);
        lockFeeBvm = next;
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
