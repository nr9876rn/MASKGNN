// ===== FILE: TTTRMNLSplitter.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title TTTRMNL Revenue Splitter
/// @notice Pull-based ERC20 splitter that enforces TTTRMNL's 50/25/25
///         revenue policy on-chain. Anyone holding TTTRMNL can call
///         `split(amount, memo)` after approving this contract; the call
///         atomically routes:
///           - 50% to 0x...dEaD (deflationary burn, since TTTRMNL has no
///             native `burn()` function)
///           - 25% to the rewards pool
///           - 25% to the treasury (takes the rounding remainder so no
///             wei is ever lost in division)
///         The contract is intentionally immutable and ownerless: there
///         are no admin keys, no upgrade paths, no fees. `flush(memo)`
///         exists only to sweep tokens accidentally sent to the splitter
///         address directly through the same split policy.
interface IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function transfer(address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract TTTRMNLSplitter {
    /// @notice The TTTRMNL ERC20 being split. Immutable.
    IERC20 public immutable token;
    /// @notice Address receiving the 25% rewards share. Immutable.
    address public immutable rewardsPool;
    /// @notice Address receiving the 25% team share. Immutable.
    address public immutable treasury;
    /// @notice Standard EVM burn sink. Tokens sent here are unrecoverable.
    address public constant DEAD = 0x000000000000000000000000000000000000dEaD;

    /// @notice Basis-point splits, fixed at deploy time. 50/25/25.
    uint256 public constant BURN_BPS = 5000;
    uint256 public constant REWARDS_BPS = 2500;
    uint256 public constant TEAM_BPS = 2500;

    /// @notice Emitted on every successful split. `memo` lets indexers
    ///         attribute revenue to a source (e.g. "premium_lifetime").
    /// @param payer The address that funded the split (msg.sender for
    ///        `split`, the splitter itself for `flush`).
    /// @param amount Total tokens split.
    /// @param burned 50% portion sent to DEAD.
    /// @param rewards 25% portion sent to rewardsPool.
    /// @param team 25% (+ rounding dust) sent to treasury.
    /// @param memo Free-form 32-byte tag (e.g. bytes32("premium_lifetime")).
    event Split(
        address indexed payer,
        uint256 amount,
        uint256 burned,
        uint256 rewards,
        uint256 team,
        bytes32 indexed memo
    );

    error ZeroAmount();
    error TransferFailed();
    error NothingToFlush();

    constructor(address _token, address _rewardsPool, address _treasury) {
        require(_token != address(0), "token=0");
        require(_rewardsPool != address(0), "rewards=0");
        require(_treasury != address(0), "treasury=0");
        token = IERC20(_token);
        rewardsPool = _rewardsPool;
        treasury = _treasury;
    }

    /// @notice Pull `amount` TTTRMNL from the caller and split it
    ///         50/25/25 in a single transaction. Caller MUST have
    ///         called `token.approve(splitter, amount)` first.
    /// @param amount Tokens to split, in base units (wei).
    /// @param memo Indexer tag. Use bytes32(0) if you don't care.
    function split(uint256 amount, bytes32 memo) external {
        if (amount == 0) revert ZeroAmount();
        // Team takes the remainder so integer-division dust never
        // strands wei. burn + rewards + team == amount exactly.
        uint256 burnAmt = (amount * BURN_BPS) / 10_000;
        uint256 rewardsAmt = (amount * REWARDS_BPS) / 10_000;
        uint256 teamAmt = amount - burnAmt - rewardsAmt;

        if (!token.transferFrom(msg.sender, DEAD, burnAmt)) revert TransferFailed();
        if (!token.transferFrom(msg.sender, rewardsPool, rewardsAmt)) revert TransferFailed();
        if (!token.transferFrom(msg.sender, treasury, teamAmt)) revert TransferFailed();

        emit Split(msg.sender, amount, burnAmt, rewardsAmt, teamAmt, memo);
    }

    /// @notice Sweep the splitter's own TTTRMNL balance through the
    ///         same 50/25/25 policy. Lets us recover from someone
    ///         doing a plain `transfer` to this address instead of
    ///         the `approve` + `split` flow.
    function flush(bytes32 memo) external {
        uint256 bal = token.balanceOf(address(this));
        if (bal == 0) revert NothingToFlush();
        uint256 burnAmt = (bal * BURN_BPS) / 10_000;
        uint256 rewardsAmt = (bal * REWARDS_BPS) / 10_000;
        uint256 teamAmt = bal - burnAmt - rewardsAmt;

        if (!token.transfer(DEAD, burnAmt)) revert TransferFailed();
        if (!token.transfer(rewardsPool, rewardsAmt)) revert TransferFailed();
        if (!token.transfer(treasury, teamAmt)) revert TransferFailed();

        emit Split(address(this), bal, burnAmt, rewardsAmt, teamAmt, memo);
    }
}
