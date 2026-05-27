// ===== FILE: FreaksAgentBindings.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title FreaksAgentBindings
/// @notice ERC-8217 binding contract for FREAKS V1 and FREAKS MUTATION V2.
/// @dev Singleton. Immutable. Ownerless. No upgrade path. Two collections only.
/// @dev Sovereignty pattern: AgentIdentity stays in the source NFT's ERC-6551 TBA.
///      The holder signs from their EOA. The contract verifies:
///        (1) msg.sender owns the source NFT
///        (2) the AgentIdentity is owned by the ERC-6551 TBA computed for that source NFT
///      No transfers required. No custody. Holder signs once.

interface IERCAgentBindings {
    enum TokenStandard { ERC721, ERC1155, ERC6909 }

    struct Binding {
        TokenStandard standard;
        address tokenContract;
        uint256 tokenId;
    }

    function bindingOf(uint256 agentId) external view returns (Binding memory);
}

interface IERC721 {
    function ownerOf(uint256 tokenId) external view returns (address);
}

interface IERC8004Registry {
    function ownerOf(uint256 agentId) external view returns (address);
}

interface IERC6551Registry {
    function account(
        address implementation,
        bytes32 salt,
        uint256 chainId,
        address tokenContract,
        uint256 tokenId
    ) external view returns (address);
}

contract FreaksAgentBindings is IERCAgentBindings {

    // ─── Collections ───────────────────────────────────────────────
    address public constant FREAKS_V1 =
        0x31Bc50C6B4B98893fD5f619Af217dBE0caAb234B;
    address public constant FREAKS_V2 =
        0x041EFE26DDfC1B446E03f68260e2621Af1C47112;

    uint256 public constant FREAKS_V1_MAX_ID = 10000;
    uint256 public constant FREAKS_V2_MAX_ID = 1111;

    // ─── ERC-8004 ──────────────────────────────────────────────────
    address public constant IDENTITY_REGISTRY =
        0x8004A169FB4a3325136EB29fA0ceB6D2e539a432;

    // ─── ERC-6551 ──────────────────────────────────────────────────
    address public constant ERC6551_REGISTRY =
        0x000000006551c19487814612e58FE06813775758;

    /// @dev The TBA implementation used by FreaksV1AgentRegistrar.
    address public constant FREAKS_V1_TBA_IMPLEMENTATION =
        0xeA61326B86531E7Ce5eDDD1F44EDf2cb12Dc5538;

    /// @dev Salt used by FreaksV1AgentRegistrar when computing TBA addresses.
    bytes32 public constant FREAKS_V1_TBA_SALT = bytes32(0);

    /// @dev FREAKS V2 will deploy its own TBA implementation. Set address here
    ///      before V2 launch. For now: same as V1, will be replaced via redeploy
    ///      if V2 ships a different TBA implementation.
    address public constant FREAKS_V2_TBA_IMPLEMENTATION =
        0xeA61326B86531E7Ce5eDDD1F44EDf2cb12Dc5538;

    bytes32 public constant FREAKS_V2_TBA_SALT = bytes32(0);

    // ─── ERC-8217 binding metadata ─────────────────────────────────
    string public constant BINDING_METADATA_KEY = "agent-binding";

    // ─── Storage ───────────────────────────────────────────────────
    mapping(uint256 => Binding) private _bindings;
    mapping(address => mapping(uint256 => uint256)) public agentIdOf;

    // ─── Events ────────────────────────────────────────────────────
    event AgentBound(
        uint256 indexed agentId,
        TokenStandard indexed standard,
        address indexed tokenContract,
        uint256 tokenId,
        address registeredBy
    );

    // ─── Errors ────────────────────────────────────────────────────
    error UnknownAgent(uint256 agentId);
    error BindingExists(uint256 agentId);
    error TokenAlreadyBound(address tokenContract, uint256 tokenId);
    error UnsupportedCollection(address tokenContract);
    error InvalidTokenId(uint256 tokenId);
    error NotTokenOwner(address caller, address tokenOwner);
    error AgentNotInExpectedTBA(address actualOwner, address expectedTBA);

    // ─── External ──────────────────────────────────────────────────

    /// @notice Register a binding between a source NFT (FREAKS V1 or V2)
    ///         and its AgentIdentity (ERC-8004 NFT).
    /// @dev    Caller must own the source NFT. The AgentIdentity must be
    ///         held by the ERC-6551 TBA computed for that source NFT.
    ///         Both checks must pass atomically.
    function registerBinding(
        uint256 agentId,
        address tokenContract,
        uint256 tokenId
    ) external {
        // (0) sanity
        if (agentId == 0) revert UnknownAgent(0);

        // (1) collection guard
        if (tokenContract != FREAKS_V1 && tokenContract != FREAKS_V2) {
            revert UnsupportedCollection(tokenContract);
        }

        // (2) tokenId range guard
        uint256 maxId = (tokenContract == FREAKS_V1)
            ? FREAKS_V1_MAX_ID
            : FREAKS_V2_MAX_ID;
        if (tokenId == 0 || tokenId > maxId) {
            revert InvalidTokenId(tokenId);
        }

        // (3) write-once on agentId
        if (_bindings[agentId].tokenContract != address(0)) {
            revert BindingExists(agentId);
        }
        // (4) write-once on (tokenContract, tokenId)
        if (agentIdOf[tokenContract][tokenId] != 0) {
            revert TokenAlreadyBound(tokenContract, tokenId);
        }

        // (5) caller owns the source NFT
        address tokenOwner = IERC721(tokenContract).ownerOf(tokenId);
        if (msg.sender != tokenOwner) {
            revert NotTokenOwner(msg.sender, tokenOwner);
        }

        // (6) AgentIdentity must be in the ERC-6551 TBA of (tokenContract, tokenId)
        address expectedTBAaddr = _computeTBA(tokenContract, tokenId);
        address agentOwner =
            IERC8004Registry(IDENTITY_REGISTRY).ownerOf(agentId);
        if (agentOwner != expectedTBAaddr) {
            revert AgentNotInExpectedTBA(agentOwner, expectedTBAaddr);
        }

        // ─── Write ─────────────────────────────────────────────────
        _bindings[agentId] = Binding({
            standard: TokenStandard.ERC721,
            tokenContract: tokenContract,
            tokenId: tokenId
        });
        agentIdOf[tokenContract][tokenId] = agentId;

        emit AgentBound(
            agentId,
            TokenStandard.ERC721,
            tokenContract,
            tokenId,
            tokenOwner
        );
    }

    /// @notice ERC-8217 forward lookup.
    function bindingOf(uint256 agentId)
        external
        view
        returns (Binding memory)
    {
        Binding memory b = _bindings[agentId];
        if (b.tokenContract == address(0)) revert UnknownAgent(agentId);
        return b;
    }

    /// @notice Helper: expected TBA address for (tokenContract, tokenId).
    ///         Useful for indexers and frontends doing pre-flight checks.
    function expectedTBA(address tokenContract, uint256 tokenId)
        external
        view
        returns (address)
    {
        if (tokenContract != FREAKS_V1 && tokenContract != FREAKS_V2) {
            revert UnsupportedCollection(tokenContract);
        }
        return _computeTBA(tokenContract, tokenId);
    }

    // ─── Internal ──────────────────────────────────────────────────

    function _computeTBA(address tokenContract, uint256 tokenId)
        internal
        view
        returns (address)
    {
        (address impl, bytes32 salt) = (tokenContract == FREAKS_V1)
            ? (FREAKS_V1_TBA_IMPLEMENTATION, FREAKS_V1_TBA_SALT)
            : (FREAKS_V2_TBA_IMPLEMENTATION, FREAKS_V2_TBA_SALT);

        return IERC6551Registry(ERC6551_REGISTRY).account(
            impl,
            salt,
            block.chainid,
            tokenContract,
            tokenId
        );
    }
}