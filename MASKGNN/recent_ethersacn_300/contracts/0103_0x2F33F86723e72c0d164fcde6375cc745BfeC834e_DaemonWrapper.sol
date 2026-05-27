// ===== FILE: Daemons/DaemonWrapper.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

interface IDAEMON {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function renderer() external view returns (address);
    function hook() external view returns (address);
}

interface IDaemonHook {
    function corruption(bytes32 poolId) external view returns (bytes32);
    function defaultPoolId() external view returns (bytes32);
}

interface IDaemonRenderer {
    function tokenURI(
        uint256 tokenId,
        bytes32 seed,
        bytes32 corruption,
        uint64  spawnBlock,
        bool    sanitized,
        bool    aura
    ) external view returns (string memory);
}

interface IERC721Receiver {
    function onERC721Received(address, address, uint256, bytes calldata) external returns (bytes4);
}

contract DaemonWrapper {
    string public constant name   = "DAEMON Frozen";
    string public constant symbol = "fDAEMON";

    address public constant DEAD          = 0x000000000000000000000000000000000000dEaD;
    uint256 public constant WRAP_AMOUNT   = 1 ether;
    uint256 public constant SANITIZE_BURN = 4 ether;
    uint256 public constant AURA_BURN     = 9 ether;

    IDAEMON public immutable daemon;

    struct Frozen {
        bytes32 seed;
        bytes32 corruption;
        uint64  spawnBlock;
        bool    sanitized;
        bool    aura;
    }
    mapping(uint256 => Frozen) public frozen;

    mapping(uint256 => address) public ownerOf;
    mapping(address => uint256) public balanceOf;
    mapping(uint256 => address) public getApproved;
    mapping(address => mapping(address => bool)) public isApprovedForAll;

    uint256 public totalMinted;
    uint256 public totalUnwrapped;
    uint256 public totalBurnedForUpgrade;

    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);
    event Approval(address indexed owner, address indexed approved, uint256 indexed tokenId);
    event ApprovalForAll(address indexed owner, address indexed operator, bool approved);
    event Wrapped(uint256 indexed tokenId, address indexed holder, bytes32 seed, bytes32 corruption, uint64 spawnBlock);
    event Unwrapped(uint256 indexed tokenId, address indexed holder);
    event Sanitized(uint256 indexed tokenId, address indexed holder);
    event Awakened(uint256 indexed tokenId, address indexed holder);

    error ZeroAddress();
    error NotOwner();
    error NotAuthorized();
    error TokenMissing();
    error AlreadySanitized();
    error AlreadyAura();
    error MustSanitizeFirst();
    error TransferFailed();
    error UnsafeReceiver();

    constructor(address _daemon) {
        if (_daemon == address(0)) revert ZeroAddress();
        daemon = IDAEMON(_daemon);
    }

    function wrap(uint256 slot) external returns (uint256 tokenId) {
        if (!daemon.transferFrom(msg.sender, address(this), WRAP_AMOUNT)) revert TransferFailed();

        bytes32 seed = keccak256(abi.encodePacked(msg.sender, slot, "DAEMON"));
        bytes32 corr;
        address h = daemon.hook();
        if (h != address(0)) {
            bytes32 poolId = IDaemonHook(h).defaultPoolId();
            corr = IDaemonHook(h).corruption(poolId);
        }

        unchecked { tokenId = ++totalMinted; }
        frozen[tokenId] = Frozen({
            seed:       seed,
            corruption: corr,
            spawnBlock: uint64(block.number),
            sanitized:  false,
            aura:       false
        });
        ownerOf[tokenId] = msg.sender;
        unchecked { balanceOf[msg.sender]++; }

        emit Transfer(address(0), msg.sender, tokenId);
        emit Wrapped(tokenId, msg.sender, seed, corr, uint64(block.number));
    }

    function unwrap(uint256 tokenId) external {
        if (ownerOf[tokenId] != msg.sender) revert NotOwner();
        delete frozen[tokenId];
        ownerOf[tokenId] = address(0);
        delete getApproved[tokenId];
        unchecked {
            balanceOf[msg.sender]--;
            totalUnwrapped++;
        }
        if (!daemon.transfer(msg.sender, WRAP_AMOUNT)) revert TransferFailed();
        emit Transfer(msg.sender, address(0), tokenId);
        emit Unwrapped(tokenId, msg.sender);
    }

    function sanitize(uint256 tokenId) external {
        if (ownerOf[tokenId] != msg.sender) revert NotOwner();
        Frozen storage f = frozen[tokenId];
        if (f.sanitized) revert AlreadySanitized();
        if (!daemon.transferFrom(msg.sender, DEAD, SANITIZE_BURN)) revert TransferFailed();
        f.sanitized = true;
        unchecked { totalBurnedForUpgrade += SANITIZE_BURN; }
        emit Sanitized(tokenId, msg.sender);
    }

    function awaken(uint256 tokenId) external {
        if (ownerOf[tokenId] != msg.sender) revert NotOwner();
        Frozen storage f = frozen[tokenId];
        if (!f.sanitized) revert MustSanitizeFirst();
        if (f.aura) revert AlreadyAura();
        if (!daemon.transferFrom(msg.sender, DEAD, AURA_BURN)) revert TransferFailed();
        f.aura = true;
        unchecked { totalBurnedForUpgrade += AURA_BURN; }
        emit Awakened(tokenId, msg.sender);
    }

    function tokenURI(uint256 tokenId) external view returns (string memory) {
        if (ownerOf[tokenId] == address(0)) revert TokenMissing();
        Frozen memory f = frozen[tokenId];
        address renderer = daemon.renderer();
        if (renderer == address(0)) return "";
        return IDaemonRenderer(renderer).tokenURI(
            tokenId, f.seed, f.corruption, f.spawnBlock, f.sanitized, f.aura
        );
    }

    function approve(address to, uint256 tokenId) external {
        address o = ownerOf[tokenId];
        if (o == address(0)) revert TokenMissing();
        if (msg.sender != o && !isApprovedForAll[o][msg.sender]) revert NotAuthorized();
        getApproved[tokenId] = to;
        emit Approval(o, to, tokenId);
    }

    function setApprovalForAll(address operator, bool approved) external {
        isApprovedForAll[msg.sender][operator] = approved;
        emit ApprovalForAll(msg.sender, operator, approved);
    }

    function transferFrom(address from, address to, uint256 tokenId) public {
        if (to == address(0)) revert ZeroAddress();
        if (ownerOf[tokenId] != from) revert NotOwner();
        if (msg.sender != from
            && getApproved[tokenId] != msg.sender
            && !isApprovedForAll[from][msg.sender]) revert NotAuthorized();
        unchecked {
            balanceOf[from]--;
            balanceOf[to]++;
        }
        ownerOf[tokenId] = to;
        delete getApproved[tokenId];
        emit Transfer(from, to, tokenId);
    }

    function safeTransferFrom(address from, address to, uint256 tokenId) external {
        safeTransferFrom(from, to, tokenId, "");
    }

    function safeTransferFrom(address from, address to, uint256 tokenId, bytes memory data) public {
        transferFrom(from, to, tokenId);
        if (to.code.length != 0) {
            try IERC721Receiver(to).onERC721Received(msg.sender, from, tokenId, data) returns (bytes4 r) {
                if (r != IERC721Receiver.onERC721Received.selector) revert UnsafeReceiver();
            } catch { revert UnsafeReceiver(); }
        }
    }

    function supportsInterface(bytes4 i) external pure returns (bool) {
        return i == 0x80ac58cd || i == 0x5b5e139f || i == 0x01ffc9a7;
    }
}
