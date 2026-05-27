// ===== FILE: contracts_-_bvm/utilities/BVMInscribe.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

interface IERC721Receiver { function onERC721Received(address, address, uint256, bytes calldata) external returns (bytes4); }

contract BVMInscribe is BVMFeeBase {
    string public constant name   = "BitcoinVM Inscriptions";
    string public constant symbol = "BVMI";

    uint256 public inscribeFeeBvm = 500 ether;

    struct Inscription {
        address author;
        uint64  timestamp;
        bytes32 contentHash;
        string  svgContent;
    }
    Inscription[] public inscriptions;

    mapping(uint256 => address) public ownerOf;
    mapping(address => uint256) public balanceOf;
    mapping(uint256 => address) public getApproved;
    mapping(address => mapping(address => bool)) public isApprovedForAll;

    event Transfer(address indexed from, address indexed to, uint256 indexed id);
    event Approval(address indexed o, address indexed approved, uint256 indexed id);
    event ApprovalForAll(address indexed o, address indexed operator, bool approved);
    event Inscribed(uint256 indexed id, address indexed author, bytes32 contentHash);
    event FeeChanged(uint256 prev, uint256 next);

    error TooLong();
    error Empty();
    error TokenMissing();

    error NotAuthorized();

    error UnsafeReceiver();

    uint256 public constant MAX_LEN = 2048;

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function inscribe(string calldata svgContent) external returns (uint256 id) {
        bytes memory b = bytes(svgContent);
        if (b.length == 0) revert Empty();
        if (b.length > MAX_LEN) revert TooLong();
        _payFee(inscribeFeeBvm);
        id = inscriptions.length + 1;
        inscriptions.push(Inscription({
            author: msg.sender, timestamp: uint64(block.timestamp),
            contentHash: keccak256(b), svgContent: svgContent
        }));
        ownerOf[id] = msg.sender;
        unchecked { balanceOf[msg.sender]++; }
        emit Transfer(address(0), msg.sender, id);
        emit Inscribed(id, msg.sender, keccak256(b));
    }

    function tokenURI(uint256 id) external view returns (string memory) {
        if (ownerOf[id] == address(0)) revert TokenMissing();
        Inscription memory i = inscriptions[id - 1];
        return string.concat(
            "data:application/json;utf8,",
            '{"name":"BVM Inscription #', _itoa(id),
            '","description":"Fully on-chain inscription minted via BitcoinVM.",',
            '"image_data":"', i.svgContent, '"}'
        );
    }

    function totalInscriptions() external view returns (uint256) { return inscriptions.length; }

    function approve(address to, uint256 id) external {
        address o = ownerOf[id];
        if (o == address(0)) revert TokenMissing();
        if (msg.sender != o && !isApprovedForAll[o][msg.sender]) revert NotAuthorized();
        getApproved[id] = to;
        emit Approval(o, to, id);
    }
    function setApprovalForAll(address op, bool ok) external {
        isApprovedForAll[msg.sender][op] = ok;
        emit ApprovalForAll(msg.sender, op, ok);
    }
    function transferFrom(address from, address to, uint256 id) public {
        if (to == address(0)) revert ZeroAddress();
        if (ownerOf[id] != from) revert NotOwner();
        if (msg.sender != from && getApproved[id] != msg.sender && !isApprovedForAll[from][msg.sender]) revert NotAuthorized();
        unchecked { balanceOf[from]--; balanceOf[to]++; }
        ownerOf[id] = to;
        delete getApproved[id];
        emit Transfer(from, to, id);
    }
    function safeTransferFrom(address from, address to, uint256 id) external {
        safeTransferFrom(from, to, id, "");
    }
    function safeTransferFrom(address from, address to, uint256 id, bytes memory data) public {
        transferFrom(from, to, id);
        if (to.code.length != 0) {
            try IERC721Receiver(to).onERC721Received(msg.sender, from, id, data) returns (bytes4 r) {
                if (r != IERC721Receiver.onERC721Received.selector) revert UnsafeReceiver();
            } catch { revert UnsafeReceiver(); }
        }
    }
    function supportsInterface(bytes4 i) external pure returns (bool) {
        return i == 0x80ac58cd || i == 0x5b5e139f || i == 0x01ffc9a7;
    }

    function setFee(uint256 next) external onlyOwner {
        emit FeeChanged(inscribeFeeBvm, next);
        inscribeFeeBvm = next;
    }

    function _itoa(uint256 v) internal pure returns (string memory) {
        if (v == 0) return "0";
        uint256 t = v; uint256 d;
        while (t != 0) { d++; t /= 10; }
        bytes memory s = new bytes(d);
        while (v != 0) { d--; s[d] = bytes1(uint8(48 + v % 10)); v /= 10; }
        return string(s);
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
