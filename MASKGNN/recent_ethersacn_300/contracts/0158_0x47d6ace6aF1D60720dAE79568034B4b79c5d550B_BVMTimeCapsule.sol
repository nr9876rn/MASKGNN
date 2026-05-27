// ===== FILE: contracts_-_bvm/utilities/BVMTimeCapsule.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

contract BVMTimeCapsule is BVMFeeBase {
    uint256 public openFeeBvm = 100 ether;

    struct Capsule {
        address creator;
        address recipient;
        uint64  unlockAt;
        bool    opened;
        bytes32 contentHash;
        string  cid;
        uint128 ethLocked;
    }
    Capsule[] public capsules;
    mapping(address => uint256[]) public capsulesOf;

    event Sealed(uint256 indexed id, address indexed creator, address indexed recipient, uint64 unlockAt, uint128 ethLocked, bytes32 contentHash, string cid);
    event Opened(uint256 indexed id, address indexed recipient, uint128 ethReleased);
    event FeeChanged(uint256 prev, uint256 next);

    error InvalidUnlock();
    error ZeroRecipient();
    error NotRecipient();
    error TooEarly();
    error AlreadyOpened();
    error EthSendFail();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function seal(address recipient, uint64 unlockAt, bytes32 contentHash, string calldata cid) external payable returns (uint256 id) {
        if (recipient == address(0)) revert ZeroRecipient();
        if (unlockAt <= block.timestamp) revert InvalidUnlock();
        _payFee(openFeeBvm);

        id = capsules.length;
        capsules.push(Capsule({
            creator:     msg.sender,
            recipient:   recipient,
            unlockAt:    unlockAt,
            opened:      false,
            contentHash: contentHash,
            cid:         cid,
            ethLocked:   uint128(msg.value)
        }));
        capsulesOf[recipient].push(id);
        emit Sealed(id, msg.sender, recipient, unlockAt, uint128(msg.value), contentHash, cid);
    }

    function open(uint256 id) external {
        Capsule storage c = capsules[id];
        if (c.opened) revert AlreadyOpened();
        if (msg.sender != c.recipient) revert NotRecipient();
        if (block.timestamp < c.unlockAt) revert TooEarly();
        c.opened = true;
        uint128 amt = c.ethLocked;
        c.ethLocked = 0;
        if (amt > 0) {
            (bool ok, ) = msg.sender.call{value: amt}("");
            if (!ok) revert EthSendFail();
        }
        emit Opened(id, msg.sender, amt);
    }

    function capsulesForRecipient(address who) external view returns (uint256[] memory) {
        return capsulesOf[who];
    }

    function totalCapsules() external view returns (uint256) { return capsules.length; }

    function setFee(uint256 next) external onlyOwner {
        emit FeeChanged(openFeeBvm, next);
        openFeeBvm = next;
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
