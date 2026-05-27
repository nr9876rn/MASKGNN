// ===== FILE: contracts_-_bvm/utilities/BVMStash.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

interface IERC20Min { function transfer(address, uint256) external returns (bool); function transferFrom(address, address, uint256) external returns (bool); }

contract BVMStash is BVMFeeBase {
    uint256 public stashFeeBvm = 210 ether;

    struct Stash {
        address holder;
        address token;
        uint128 amount;
        uint64  delaySecs;
        uint64  requestedAt;
        bool    withdrawn;
    }
    Stash[] public stashes;
    mapping(address => uint256[]) public stashesOf;

    event Stashed(uint256 indexed id, address indexed holder, address indexed token, uint128 amount, uint64 delaySecs);
    event WithdrawRequested(uint256 indexed id, uint64 unlocksAt);
    event Withdrawn(uint256 indexed id, uint128 amount);
    event FeeChanged(uint256 prev, uint256 next);

    error BadParams();
    error NotHolder();
    error AlreadyOut();
    error NotRequested();
    error TooEarly();
    error TransferFailed();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function createEth(uint64 delaySecs) external payable returns (uint256 id) {
        if (msg.value == 0 || delaySecs < 1 hours) revert BadParams();
        _payFee(stashFeeBvm);
        id = stashes.length;
        stashes.push(Stash({
            holder: msg.sender, token: address(0),
            amount: uint128(msg.value), delaySecs: delaySecs,
            requestedAt: 0, withdrawn: false
        }));
        stashesOf[msg.sender].push(id);
        emit Stashed(id, msg.sender, address(0), uint128(msg.value), delaySecs);
    }

    function createToken(address token, uint128 amount, uint64 delaySecs) external returns (uint256 id) {
        if (token == address(0) || amount == 0 || delaySecs < 1 hours) revert BadParams();
        _payFee(stashFeeBvm);
        if (!IERC20Min(token).transferFrom(msg.sender, address(this), amount)) revert TransferFailed();
        id = stashes.length;
        stashes.push(Stash({
            holder: msg.sender, token: token,
            amount: amount, delaySecs: delaySecs,
            requestedAt: 0, withdrawn: false
        }));
        stashesOf[msg.sender].push(id);
        emit Stashed(id, msg.sender, token, amount, delaySecs);
    }

    function requestWithdraw(uint256 id) external {
        Stash storage s = stashes[id];
        if (msg.sender != s.holder) revert NotHolder();
        if (s.withdrawn) revert AlreadyOut();
        s.requestedAt = uint64(block.timestamp);
        emit WithdrawRequested(id, uint64(block.timestamp) + s.delaySecs);
    }

    function withdraw(uint256 id) external {
        Stash storage s = stashes[id];
        if (msg.sender != s.holder) revert NotHolder();
        if (s.withdrawn) revert AlreadyOut();
        if (s.requestedAt == 0) revert NotRequested();
        if (block.timestamp < uint256(s.requestedAt) + uint256(s.delaySecs)) revert TooEarly();
        s.withdrawn = true;
        if (s.token == address(0)) {
            (bool ok, ) = s.holder.call{value: s.amount}("");
            require(ok, "eth");
        } else {
            if (!IERC20Min(s.token).transfer(s.holder, s.amount)) revert TransferFailed();
        }
        emit Withdrawn(id, s.amount);
    }

    function totalStashes() external view returns (uint256) { return stashes.length; }

    function setFee(uint256 next) external onlyOwner {
        emit FeeChanged(stashFeeBvm, next);
        stashFeeBvm = next;
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
