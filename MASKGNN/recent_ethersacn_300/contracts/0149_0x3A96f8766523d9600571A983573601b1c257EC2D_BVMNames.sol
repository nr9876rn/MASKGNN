// ===== FILE: contracts_-_bvm/utilities/BVMNames.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

contract BVMNames is BVMFeeBase {
    string  public constant TLD = ".bvm";
    uint256 public claimFeeBvm = 210 ether;
    uint8   public constant MIN_LEN = 3;
    uint8   public constant MAX_LEN = 32;

    mapping(string  => address) public addressOfName;
    mapping(address => string)  public nameOfAddress;
    mapping(string  => uint64)  public claimedAt;
    uint256 public totalNames;

    event Claimed(string indexed name, address indexed who, uint256 burnedBvm);
    event Released(string indexed name, address indexed prior);
    event Transferred(string indexed name, address indexed from, address indexed to);
    event FeeChanged(uint256 prev, uint256 next);

    error InvalidName();
    error TooShort();
    error TooLong();
    error NameTaken();
    error AlreadyHasName();
    error NotNameOwner();
    error ZeroTo();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function isAvailable(string calldata n) external view returns (bool) {
        bytes memory b = bytes(n);
        if (b.length < MIN_LEN || b.length > MAX_LEN) return false;
        if (!_valid(b)) return false;
        return addressOfName[n] == address(0);
    }

    function claim(string calldata n) external {
        bytes memory b = bytes(n);
        if (b.length < MIN_LEN) revert TooShort();
        if (b.length > MAX_LEN) revert TooLong();
        if (!_valid(b)) revert InvalidName();
        if (addressOfName[n] != address(0)) revert NameTaken();
        if (bytes(nameOfAddress[msg.sender]).length != 0) revert AlreadyHasName();

        _payFee(claimFeeBvm);

        addressOfName[n] = msg.sender;
        nameOfAddress[msg.sender] = n;
        claimedAt[n] = uint64(block.timestamp);
        unchecked { totalNames++; }
        emit Claimed(n, msg.sender, claimFeeBvm);
    }

    function release(string calldata n) external {
        if (addressOfName[n] != msg.sender) revert NotNameOwner();
        delete addressOfName[n];
        delete nameOfAddress[msg.sender];
        delete claimedAt[n];
        unchecked { totalNames--; }
        emit Released(n, msg.sender);
    }

    function transferName(string calldata n, address to) external {
        if (addressOfName[n] != msg.sender) revert NotNameOwner();
        if (to == address(0)) revert ZeroTo();
        if (bytes(nameOfAddress[to]).length != 0) revert AlreadyHasName();
        addressOfName[n] = to;
        delete nameOfAddress[msg.sender];
        nameOfAddress[to] = n;
        emit Transferred(n, msg.sender, to);
    }

    function resolve(string calldata n) external view returns (address) {
        return addressOfName[n];
    }

    function reverse(address who) external view returns (string memory) {
        return nameOfAddress[who];
    }

    function setFee(uint256 next) external onlyOwner {
        emit FeeChanged(claimFeeBvm, next);
        claimFeeBvm = next;
    }

    function _valid(bytes memory b) internal pure returns (bool) {
        for (uint256 i; i < b.length; i++) {
            bytes1 c = b[i];
            bool ok = (c >= 0x61 && c <= 0x7A) || (c >= 0x30 && c <= 0x39) || c == 0x5F || c == 0x2D;
            if (!ok) return false;
        }
        return true;
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
