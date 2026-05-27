// ===== FILE: contracts_-_bvm/utilities/BVMEpitaph.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

contract BVMEpitaph is BVMFeeBase {
    uint256 public inscribeFeeBvm = 500 ether;
    uint256 public constant MAX_LEN = 280;

    struct Inscription {
        address author;
        uint64  timestamp;
        string  message;
    }
    Inscription[] public inscriptions;
    mapping(address => uint256[]) public inscriptionsOf;

    event Inscribed(uint256 indexed id, address indexed author, string message, uint256 feeBvm);
    event FeeChanged(uint256 prev, uint256 next);

    error TooLong();
    error Empty();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function inscribe(string calldata message) external returns (uint256 id) {
        bytes memory b = bytes(message);
        if (b.length == 0) revert Empty();
        if (b.length > MAX_LEN) revert TooLong();
        _payFee(inscribeFeeBvm);

        id = inscriptions.length;
        inscriptions.push(Inscription({
            author:    msg.sender,
            timestamp: uint64(block.timestamp),
            message:   message
        }));
        inscriptionsOf[msg.sender].push(id);
        emit Inscribed(id, msg.sender, message, inscribeFeeBvm);
    }

    function totalInscriptions() external view returns (uint256) { return inscriptions.length; }

    function recent(uint256 count) external view returns (Inscription[] memory out) {
        uint256 n = inscriptions.length;
        if (count > n) count = n;
        out = new Inscription[](count);
        for (uint256 i; i < count; i++) {
            out[i] = inscriptions[n - 1 - i];
        }
    }

    function inscriptionsForAuthor(address who) external view returns (uint256[] memory) {
        return inscriptionsOf[who];
    }

    function setFee(uint256 next) external onlyOwner {
        emit FeeChanged(inscribeFeeBvm, next);
        inscribeFeeBvm = next;
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
