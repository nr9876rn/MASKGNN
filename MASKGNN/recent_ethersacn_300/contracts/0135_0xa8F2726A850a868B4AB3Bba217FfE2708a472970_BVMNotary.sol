// ===== FILE: contracts_-_bvm/utilities/BVMNotary.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

contract BVMNotary is BVMFeeBase {
    uint256 public stampFeeBvm = 50 ether;

    struct Stamp {
        address notary;
        uint64  timestamp;
        bytes32 hash;
        string  note;
    }
    Stamp[] public stamps;
    mapping(bytes32 => uint256) public stampOf;
    mapping(address => uint256[]) public stampsBy;

    event Stamped(uint256 indexed id, address indexed notary, bytes32 indexed hash, uint64 timestamp, string note);
    event FeeChanged(uint256 prev, uint256 next);

    error AlreadyStamped();
    error EmptyHash();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function stamp(bytes32 hash, string calldata note) external returns (uint256 id) {
        if (hash == bytes32(0)) revert EmptyHash();
        if (stampOf[hash] != 0) revert AlreadyStamped();
        _payFee(stampFeeBvm);
        id = stamps.length;
        stamps.push(Stamp({
            notary: msg.sender, timestamp: uint64(block.timestamp),
            hash: hash, note: note
        }));
        stampOf[hash] = id + 1;
        stampsBy[msg.sender].push(id);
        emit Stamped(id, msg.sender, hash, uint64(block.timestamp), note);
    }

    function isStamped(bytes32 hash) external view returns (bool) {
        return stampOf[hash] != 0;
    }

    function lookup(bytes32 hash) external view returns (address notary, uint64 timestamp, string memory note) {
        uint256 idx = stampOf[hash];
        if (idx == 0) return (address(0), 0, "");
        Stamp storage s = stamps[idx - 1];
        return (s.notary, s.timestamp, s.note);
    }

    function totalStamps() external view returns (uint256) { return stamps.length; }

    function setFee(uint256 next) external onlyOwner {
        emit FeeChanged(stampFeeBvm, next);
        stampFeeBvm = next;
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
