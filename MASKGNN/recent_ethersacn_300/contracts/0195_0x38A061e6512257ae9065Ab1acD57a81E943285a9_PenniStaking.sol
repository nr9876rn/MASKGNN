// ===== FILE: penni/PenniStaking2.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "https://github.com/OpenZeppelin/openzeppelin-contracts/blob/v4.9.6/contracts/security/ReentrancyGuard.sol";

interface IERC721 {
    function ownerOf(uint256 tokenId) external view returns (address);
    function transferFrom(address from, address to, uint256 tokenId) external;
    function safeTransferFrom(address from, address to, uint256 tokenId) external;
    function isApprovedForAll(address owner, address operator) external view returns (bool);
}

/// @title PenniStaking v3 - custodial fixed-term locks
/// @notice Custodial. NFTs are transferred to this contract on stake() and
///         returned on unstake(). This prevents marketplace listings while
///         staked - protecting floor price. Holders pick a lock period at
///         stake time; NFTs become unstakeable once `unlockAt` is reached.
///         Owner can flip a one-way emergency switch that lets every staker
///         withdraw immediately, guaranteeing no NFT can ever be trapped.
contract PenniStaking is ReentrancyGuard {
    IERC721 public immutable nft;
    address public immutable owner;

    /// Fixed lock periods, indexed by tier (0..4)
    uint64[5] public LOCK_DURATIONS = [
        uint64(7 days),
        uint64(15 days),
        uint64(30 days),
        uint64(90 days),
        uint64(180 days)
    ];

    /// One-way emergency switch. Once true, any staker can unstake at any time.
    bool public emergencyUnlock;

    /// On-chain kill switch for new stakes. Owner-controlled, two-way.
    /// While false, stake() reverts - preventing direct Etherscan abuse before launch.
    /// unstake() is intentionally always allowed so NFTs are never trapped.
    bool public stakingEnabled;

    struct StakeInfo {
        address owner;
        uint64  stakedAt;
        uint64  unlockAt;
        uint8   tier;
    }

    mapping(uint256 => StakeInfo) public stakes;

    event Staked(address indexed user, uint256 indexed tokenId, uint8 tier, uint64 stakedAt, uint64 unlockAt);
    event Unstaked(address indexed user, uint256 indexed tokenId, uint64 timestamp, bool early);
    event EmergencyUnlockEnabled(uint64 timestamp);
    event StakingEnabledSet(bool enabled);

    error NotOwner();
    error AlreadyStaked();
    error NotStaker();
    error BadBatch();
    error BadTier();
    error StillLocked();
    error OnlyOwner();
    error AlreadyEnabled();
    error StakingDisabled();
    error NotApproved();

    constructor(address _nft) {
        require(_nft != address(0), "nft=0");
        nft   = IERC721(_nft);
        owner = msg.sender;
    }

    /// @notice Stake one or more Penni for the given tier. Requires the user
    ///         to have called nft.setApprovalForAll(stakingContract, true) once.
    function stake(uint256[] calldata tokenIds, uint8 tier) external nonReentrant {
        if (!stakingEnabled) revert StakingDisabled();
        uint256 len = tokenIds.length;
        if (len == 0 || len > 50) revert BadBatch();
        if (tier > 4) revert BadTier();
        if (!nft.isApprovedForAll(msg.sender, address(this))) revert NotApproved();

        uint64 nowTs   = uint64(block.timestamp);
        uint64 unlock_ = nowTs + LOCK_DURATIONS[tier];

        for (uint256 i; i < len; ++i) {
            uint256 id = tokenIds[i];
            if (nft.ownerOf(id) != msg.sender) revert NotOwner();
            if (stakes[id].owner != address(0)) revert AlreadyStaked();

            stakes[id] = StakeInfo({
                owner:    msg.sender,
                stakedAt: nowTs,
                unlockAt: unlock_,
                tier:     tier
            });

            // Pull the NFT into the staking contract (custodial)
            nft.transferFrom(msg.sender, address(this), id);
            emit Staked(msg.sender, id, tier, nowTs, unlock_);
        }
    }

    /// @notice Unstake one or more Penni. Returns NFTs to the original staker.
    function unstake(uint256[] calldata tokenIds) external nonReentrant {
        uint256 len = tokenIds.length;
        if (len == 0 || len > 50) revert BadBatch();
        bool emergency = emergencyUnlock;
        uint64 nowTs   = uint64(block.timestamp);

        for (uint256 i; i < len; ++i) {
            uint256 id = tokenIds[i];
            StakeInfo memory s = stakes[id];
            if (s.owner != msg.sender) revert NotStaker();
            bool early = nowTs < s.unlockAt;
            if (early && !emergency) revert StillLocked();

            delete stakes[id];
            // Return the NFT to the original staker
            nft.transferFrom(address(this), s.owner, id);
            emit Unstaked(msg.sender, id, nowTs, early);
        }
    }

    /// @notice One-way kill switch. After this, every staker can withdraw at any time.
    ///         Cannot be turned off. Designed for contract migration / bug response.
    function enableEmergencyUnlock() external {
        if (msg.sender != owner) revert OnlyOwner();
        if (emergencyUnlock) revert AlreadyEnabled();
        emergencyUnlock = true;
        emit EmergencyUnlockEnabled(uint64(block.timestamp));
    }

    /// @notice Owner-only toggle that controls whether new stakes are accepted.
    function setStakingEnabled(bool enabled) external {
        if (msg.sender != owner) revert OnlyOwner();
        stakingEnabled = enabled;
        emit StakingEnabledSet(enabled);
    }

    /// @notice Emergency admin function: force-unstake specific tokenIds and
    ///         return them to their original staker, regardless of lock period.
    ///         Use only for stuck NFTs, migrations, or incident response.
    function adminForceUnstake(uint256[] calldata tokenIds) external nonReentrant {
        if (msg.sender != owner) revert OnlyOwner();
        uint256 len = tokenIds.length;
        if (len == 0 || len > 50) revert BadBatch();
        uint64 nowTs = uint64(block.timestamp);

        for (uint256 i; i < len; ++i) {
            uint256 id = tokenIds[i];
            StakeInfo memory s = stakes[id];
            if (s.owner == address(0)) continue; // skip not-staked

            delete stakes[id];
            nft.transferFrom(address(this), s.owner, id);
            emit Unstaked(s.owner, id, nowTs, nowTs < s.unlockAt);
        }
    }

    /// @notice TEST/ADMIN ONLY: shorten (or extend) the unlockAt timestamp of
    ///         given stakes. Lets the team verify "unlock period elapsed" UX
    ///         without waiting days. Does NOT move funds. Emits no event.
    function adminSetUnlockAt(uint256[] calldata tokenIds, uint64 newUnlockAt) external {
        if (msg.sender != owner) revert OnlyOwner();
        uint256 len = tokenIds.length;
        if (len == 0 || len > 50) revert BadBatch();
        for (uint256 i; i < len; ++i) {
            uint256 id = tokenIds[i];
            if (stakes[id].owner == address(0)) continue;
            stakes[id].unlockAt = newUnlockAt;
        }
    }

    function isStaked(uint256 tokenId) external view returns (bool) {
        return stakes[tokenId].owner != address(0);
    }

    function isMatured(uint256 tokenId) external view returns (bool) {
        StakeInfo memory s = stakes[tokenId];
        if (s.owner == address(0)) return false;
        return emergencyUnlock || block.timestamp >= s.unlockAt;
    }

    function stakerOf(uint256 tokenId) external view returns (address) {
        return stakes[tokenId].owner;
    }

    /// @notice ERC721Receiver - required so safeTransferFrom into this contract works
    ///         (some wallets/marketplaces use safe variant). We accept any inbound
    ///         NFT call; stake bookkeeping only happens through stake().
    function onERC721Received(address, address, uint256, bytes calldata) external pure returns (bytes4) {
        return this.onERC721Received.selector;
    }
}

// ===== FILE: https_/github.com/OpenZeppelin/openzeppelin-contracts/blob/v4.9.6/contracts/security/ReentrancyGuard.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.0) (security/ReentrancyGuard.sol)

pragma solidity ^0.8.0;

/**
 * @dev Contract module that helps prevent reentrant calls to a function.
 *
 * Inheriting from `ReentrancyGuard` will make the {nonReentrant} modifier
 * available, which can be applied to functions to make sure there are no nested
 * (reentrant) calls to them.
 *
 * Note that because there is a single `nonReentrant` guard, functions marked as
 * `nonReentrant` may not call one another. This can be worked around by making
 * those functions `private`, and then adding `external` `nonReentrant` entry
 * points to them.
 *
 * TIP: If you would like to learn more about reentrancy and alternative ways
 * to protect against it, check out our blog post
 * https://blog.openzeppelin.com/reentrancy-after-istanbul/[Reentrancy After Istanbul].
 */
abstract contract ReentrancyGuard {
    // Booleans are more expensive than uint256 or any type that takes up a full
    // word because each write operation emits an extra SLOAD to first read the
    // slot's contents, replace the bits taken up by the boolean, and then write
    // back. This is the compiler's defense against contract upgrades and
    // pointer aliasing, and it cannot be disabled.

    // The values being non-zero value makes deployment a bit more expensive,
    // but in exchange the refund on every call to nonReentrant will be lower in
    // amount. Since refunds are capped to a percentage of the total
    // transaction's gas, it is best to keep them low in cases like this one, to
    // increase the likelihood of the full refund coming into effect.
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;

    uint256 private _status;

    constructor() {
        _status = _NOT_ENTERED;
    }

    /**
     * @dev Prevents a contract from calling itself, directly or indirectly.
     * Calling a `nonReentrant` function from another `nonReentrant`
     * function is not supported. It is possible to prevent this from happening
     * by making the `nonReentrant` function external, and making it call a
     * `private` function that does the actual work.
     */
    modifier nonReentrant() {
        _nonReentrantBefore();
        _;
        _nonReentrantAfter();
    }

    function _nonReentrantBefore() private {
        // On the first call to nonReentrant, _status will be _NOT_ENTERED
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");

        // Any calls to nonReentrant after this point will fail
        _status = _ENTERED;
    }

    function _nonReentrantAfter() private {
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _status = _NOT_ENTERED;
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        return _status == _ENTERED;
    }
}
