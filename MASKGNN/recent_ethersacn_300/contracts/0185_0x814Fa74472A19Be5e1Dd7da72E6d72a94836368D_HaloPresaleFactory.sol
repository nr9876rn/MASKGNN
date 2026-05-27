// ===== FILE: HaloPresale.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

interface IUniswapV2Router02 {
    function addLiquidityETH(
        address token,
        uint amountTokenDesired,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external payable returns (uint amountToken, uint amountETH, uint liquidity);

    function factory() external pure returns (address);
    function WETH() external pure returns (address);
}

interface IUniswapV2Factory {
    function getPair(address tokenA, address tokenB) external view returns (address pair);
}

// Minimal interface for PinkLock V2 (BSC: 0x407993575c91ce7643a4d4cCACc9A98c36eE1BBE,
// Polygon: 0x3eF7442dF454bA6b7C1deEc8DdF29Cfb2d6e56c7).
// Caller must approve the locker for `amount` BEFORE calling lock().
interface IPinkLock {
    function lock(
        address owner,
        address token,
        bool isLpToken,
        uint256 amount,
        uint256 unlockDate,
        string memory description
    ) external returns (uint256 id);
}

// Minimal interface for HaloLPTimelock (per-deposit, per-withdrawer
// non-custodial LP locker). Caller must approve the locker for `amount`
// BEFORE calling deposit().
interface IHaloLPTimelock {
    function deposit(
        address token,
        uint256 amount,
        address withdrawer
    ) external returns (uint256 depositIndex);
}

// Minimal interface to flip the bound BEP20Token's trading switch.
// Used by `openTrading()` after the per-presale claim window expires.
interface IHaloToken {
    function enableTrading() external;
    function tradingEnabled() external view returns (bool);
    function presaleContract() external view returns (address);
    function platformFeeWallet() external view returns (address);
}

// Locker calling convention. Stored as uint8 in immutable factory state
// and the per-presale state to keep gas overhead at zero for the common
// path.
//   0 = RAW_TRANSFER       Legacy / TestLocker. Locker registers a lock
//                          when it receives a raw `IERC20.transfer(...)`.
//                          Custodial — withdrawer is the locker's owner,
//                          NOT the presale creator. Kept only for
//                          backward-compatibility with already-deployed
//                          factories; new factories should use kind=2.
//   1 = PINKLOCK_V2        PinkLock V2 (BSC + Polygon). Caller approves +
//                          calls lock(); PinkLock pulls via transferFrom.
//                          Lock owner is the presale creator.
//   2 = HALO_TIMELOCK_V2   HaloLPTimelock (Base + Ethereum). Caller
//                          approves + calls deposit(token, amount,
//                          withdrawer); the timelock pulls via
//                          transferFrom and records the presale creator
//                          as the sole withdrawer. Non-custodial.

contract HaloPresale is ReentrancyGuard {
    using SafeERC20 for IERC20;
    struct PresaleInfo {
        address creator;
        address token;
        uint256 hardCap;
        uint256 softCap;
        uint256 presaleRate;
        uint256 listingRate;
        uint256 minContribution;
        uint256 maxContribution;
        uint256 liquidityPercent;
        uint256 liquidityLockDays;
        uint256 startTime;
        uint256 endTime;
        uint256 totalRaised;
        uint256 participantCount;
        bool finalized;
        bool cancelled;
        string metadataURI;
    }

    PresaleInfo public presaleInfo;
    address public factory;
    address public platformWallet;
    address public router;
    address public lpLocker;
    address public lpToken;

    uint8 public lockerKind;     // 0=RAW_TRANSFER, 1=PINKLOCK_V2, 2=HALO_TIMELOCK_V2
    uint256 public lpLockId;     // 0 if not applicable; PinkLock returns a numeric id

    uint256 public tokensForPresale;
    uint256 public tokensForLiquidity;

    // === Two-phase launch (Task #29) ============================
    // Trading on the BEP20 token stays disabled until openTrading()
    // is called. claimWindow is the seconds AFTER finalize() during
    // which only the creator can call openTrading(); after the
    // window AND a 24h public failsafe, ANYONE may flip trading on
    // so a missing/buggy creator can't strand contributors.
    uint256 public claimWindow;       // seconds (1800 - 86400, default 1800)
    uint256 public tradingOpensAt;    // block.timestamp at which creator can openTrading
    bool public tradingOpened;        // mirrors token.tradingEnabled() once flipped via this wrapper
    uint256 public constant PUBLIC_OPEN_FAILSAFE = 24 hours;
    // ===========================================================

    // === Creator-never-finalize failsafe (Task #43 / Audit #2) ===
    // If the soft cap is met but the creator never calls finalize(),
    // contributors would otherwise be stuck: claim() requires
    // finalized=true and withdrawRefund() only opens on cancel OR
    // (softCap NOT met after endTime). After this grace window past
    // endTime, withdrawRefund() unlocks unconditionally so funds can
    // never be permanently trapped by an absent creator.
    uint256 public constant FORCE_REFUND_GRACE = 7 days;
    // ============================================================

    mapping(address => uint256) public contributions;
    mapping(address => bool) public hasContributed;
    mapping(address => bool) public hasClaimed;

    // Standard burn sink. Tokens sent here cannot be recovered and most
    // explorers / scanners treat balances at 0x...dEaD as "burned" for
    // circulating-supply math, which clears "owner concentration" risk
    // flags on PancakeSwap / DexTools.
    address internal constant BURN_ADDRESS = 0x000000000000000000000000000000000000dEaD;

    event Contributed(address indexed user, uint256 amount, uint256 totalRaised);
    event Refunded(address indexed user, uint256 amount);
    event Claimed(address indexed user, uint256 amount);
    event Finalized(uint256 totalRaised, uint256 liquidityBNB, uint256 liquidityTokens);
    event Cancelled();
    event LiquidityLocked(address lpToken, uint256 amount, uint256 unlockTime);
    // Emitted on finalize() once the creator-residual + any unsold-presale /
    // unused-liquidity tokens are sent to BURN_ADDRESS. Mirrors the main
    // launcher's invariant that the creator wallet ends at ~zero after
    // launch so token scanners don't flag owner concentration.
    event ExcessBurned(uint256 unsoldAndUnusedAmount, uint256 creatorResidualAmount);
    // New event: lets indexers (and the UI) discover lock kind + id without
    // re-parsing the locker's own event topology.
    event LockRegistered(uint8 indexed kind, address indexed locker, uint256 lockId);

    // Two-phase launch events.
    event ClaimWindowOpened(uint256 opensAt, uint256 claimWindow);
    event TradingOpened(address indexed by, bool publicFailsafe);

    constructor() {
        factory = address(1);
    }

    function initialize(
        address _creator,
        address _token,
        uint256[11] memory _params,
        string memory _metadataURI,
        address _platformWallet,
        address _router,
        address _lpLocker,
        uint8 _lockerKind
    ) external {
        require(factory == address(0), "Already initialized");
        require(_lockerKind <= 2, "Bad locker kind");

        require(_params[0] >= _params[1], "Hard cap must be >= soft cap");
        require(_params[1] > 0, "Soft cap must be > 0");
        require(_params[6] >= 5100 && _params[6] <= 10000, "Liquidity must be 51-100%");
        require(_params[8] >= block.timestamp, "Start time must be future");
        require(_params[9] > _params[8], "End must be after start");
        require(_params[9] <= _params[8] + 30 days, "Max duration 30 days");
        require(_params[7] >= 30, "Min 30 days LP lock");
        // Claim window: 30 minutes (1800s) — 24 hours (86400s).
        require(_params[10] >= 1800 && _params[10] <= 86400, "Claim window 30min-24h");

        tokensForPresale = (_params[0] * _params[2]) / 1e18;
        tokensForLiquidity = (_params[0] * _params[6] * _params[3]) / (1e18 * 10000);

        uint256 totalRequired = tokensForPresale + tokensForLiquidity;

        require(
            IERC20(_token).balanceOf(address(this)) >= totalRequired,
            "Tokens not received from factory"
        );

        factory = msg.sender;
        platformWallet = _platformWallet;
        router = _router;
        lpLocker = _lpLocker;
        lockerKind = _lockerKind;
        claimWindow = _params[10];

        presaleInfo = PresaleInfo({
            creator: _creator,
            token: _token,
            hardCap: _params[0],
            softCap: _params[1],
            presaleRate: _params[2],
            listingRate: _params[3],
            minContribution: _params[4],
            maxContribution: _params[5],
            liquidityPercent: _params[6],
            liquidityLockDays: _params[7],
            startTime: _params[8],
            endTime: _params[9],
            totalRaised: 0,
            participantCount: 0,
            finalized: false,
            cancelled: false,
            metadataURI: _metadataURI
        });
    }

    function contribute() external payable nonReentrant {
        require(block.timestamp >= presaleInfo.startTime, "Not started");
        require(block.timestamp < presaleInfo.endTime, "Ended");
        require(!presaleInfo.cancelled, "Cancelled");
        require(!presaleInfo.finalized, "Finalized");
        require(msg.value >= presaleInfo.minContribution, "Below min");
        require(
            contributions[msg.sender] + msg.value <= presaleInfo.maxContribution,
            "Exceeds max"
        );
        require(
            presaleInfo.totalRaised + msg.value <= presaleInfo.hardCap,
            "Exceeds hard cap"
        );

        if (!hasContributed[msg.sender]) {
            hasContributed[msg.sender] = true;
            presaleInfo.participantCount++;
        }

        contributions[msg.sender] += msg.value;
        presaleInfo.totalRaised += msg.value;

        emit Contributed(msg.sender, msg.value, presaleInfo.totalRaised);
    }

    function finalize() external nonReentrant {
        // Access control with creator-vanished failsafe (Task #43).
        //   - Creator may call any time after endTime if softCap met.
        //   - ANY address may call after endTime + FORCE_REFUND_GRACE
        //     (7 days) if softCap met. This prevents a delinquent
        //     creator from indefinitely blocking finalize and starving
        //     contributors of their claim path. The remaining-ETH
        //     payout at the end of this function still goes to
        //     `presaleInfo.creator`, so a non-creator caller has no
        //     economic incentive to grief — they just spend the gas.
        require(!presaleInfo.finalized, "Already finalized");
        require(!presaleInfo.cancelled, "Cancelled");
        require(block.timestamp >= presaleInfo.endTime, "Not ended");
        require(presaleInfo.totalRaised >= presaleInfo.softCap, "Soft cap not met");
        if (msg.sender != presaleInfo.creator) {
            require(
                block.timestamp > presaleInfo.endTime + FORCE_REFUND_GRACE,
                "Only creator before failsafe"
            );
        }

        presaleInfo.finalized = true;

        uint256 platformFee = (presaleInfo.totalRaised * 200) / 10000;
        uint256 liquidityBNB = (presaleInfo.totalRaised * presaleInfo.liquidityPercent) / 10000;
        uint256 liquidityTokensDesired = (liquidityBNB * presaleInfo.listingRate) / 1e18;

        payable(platformWallet).transfer(platformFee);

        // forceApprove handles non-standard tokens that require a
        // 0-reset between non-zero approvals (e.g. USDT-style ERC20s).
        IERC20(presaleInfo.token).forceApprove(router, liquidityTokensDesired);

        IUniswapV2Router02 uniRouter = IUniswapV2Router02(router);
        // Slippage floor: accept up to 1% deviation in either leg. The
        // wrapper is the *only* liquidity provider at this point so
        // there is no meaningful pool to slip against, but a defensive
        // minimum catches buggy/exotic routers that would otherwise
        // happily consume 0 tokens or 0 ETH on a misconfigured pair.
        // (Audit finding #4.)
        uint256 minTokens = (liquidityTokensDesired * 99) / 100;
        uint256 minETH = (liquidityBNB * 99) / 100;
        (uint256 tokensUsed, uint256 ethUsed, uint256 liquidity) = uniRouter.addLiquidityETH{value: liquidityBNB}(
            presaleInfo.token,
            liquidityTokensDesired,
            minTokens,
            minETH,
            address(this),
            block.timestamp + 1200
        );

        lpToken = IUniswapV2Factory(uniRouter.factory()).getPair(
            presaleInfo.token,
            uniRouter.WETH()
        );

        uint256 lockDuration = presaleInfo.liquidityLockDays * 1 days;
        uint256 unlockTime = block.timestamp + lockDuration;

        // Branch by locker calling convention. The owner of the lock is
        // ALWAYS the presale creator — never the platform or this
        // contract — so the platform cannot rugpull LP at unlock.
        if (lockerKind == 1) {
            // PinkLock V2: approve, then call lock(). PinkLock pulls via
            // transferFrom, emits its own Lock event (which DexScreener /
            // DexTools / GeckoTerminal index), and returns a numeric lock id.
            IERC20(lpToken).forceApprove(lpLocker, liquidity);
            uint256 id = IPinkLock(lpLocker).lock(
                presaleInfo.creator,
                lpToken,
                true,
                liquidity,
                unlockTime,
                presaleInfo.metadataURI
            );
            lpLockId = id;
            emit LockRegistered(1, lpLocker, id);
        } else if (lockerKind == 2) {
            // HaloLPTimelock (per-deposit, per-withdrawer, non-custodial).
            // Approve and call deposit() naming the presale creator as
            // the sole withdrawer. The platform never has withdraw rights
            // over locked LP.
            IERC20(lpToken).forceApprove(lpLocker, liquidity);
            uint256 depositIndex = IHaloLPTimelock(lpLocker).deposit(
                lpToken,
                liquidity,
                presaleInfo.creator
            );
            lpLockId = depositIndex;
            emit LockRegistered(2, lpLocker, depositIndex);
        } else {
            // RAW_TRANSFER: legacy. The locker registers a lock when it
            // receives the raw transfer. Withdrawer is the locker's
            // global owner (custodial — not used by new factories).
            IERC20(lpToken).safeTransfer(lpLocker, liquidity);
            emit LockRegistered(0, lpLocker, 0);
        }

        emit LiquidityLocked(lpToken, liquidity, unlockTime);

        uint256 soldTokens = (presaleInfo.totalRaised * presaleInfo.presaleRate) / 1e18;
        uint256 unsoldPresaleTokens = tokensForPresale - soldTokens;
        uint256 unusedLiquidityTokens = tokensForLiquidity > tokensUsed ? tokensForLiquidity - tokensUsed : 0;
        uint256 totalUnused = unsoldPresaleTokens + unusedLiquidityTokens;

        // Mirror the main launcher invariant: any tokens not actually
        // placed into LP or sold to contributors are burned, and any
        // residual still sitting in the creator wallet (bounded by
        // their allowance to this wrapper) is also burned. See
        // _burnExcess for details. Extracted into a helper to keep
        // finalize() under the via-IR stack limit.
        _burnExcess(totalUnused);

        uint256 remaining = presaleInfo.totalRaised - platformFee - ethUsed;
        payable(presaleInfo.creator).transfer(remaining);

        // Two-phase launch: kick off the claim window. Trading stays
        // OFF on the bound token until openTrading() is called after
        // `tradingOpensAt`. Contributors can claim() in the meantime
        // because the wrapper is exempt from the token's pre-trading
        // guard (set via setPresaleContract before factory.createPresale).
        tradingOpensAt = block.timestamp + claimWindow;
        emit Finalized(presaleInfo.totalRaised, liquidityBNB, tokensUsed);
        emit ClaimWindowOpened(tradingOpensAt, claimWindow);
    }

    function cancelPresale() external nonReentrant {
        require(msg.sender == presaleInfo.creator || msg.sender == factory, "Unauthorized");
        require(!presaleInfo.finalized, "Already finalized");
        require(!presaleInfo.cancelled, "Already cancelled");

        presaleInfo.cancelled = true;

        uint256 tokenBalance = IERC20(presaleInfo.token).balanceOf(address(this));
        if (tokenBalance > 0) {
            IERC20(presaleInfo.token).safeTransfer(presaleInfo.creator, tokenBalance);
        }

        emit Cancelled();
    }

    function withdrawRefund() external nonReentrant {
        // Refunds open in three cases:
        //   1. Presale was explicitly cancelled.
        //   2. Presale ended and the soft cap was not met (classic failure).
        //   3. Creator-never-finalize failsafe: presale ended, was NOT
        //      finalized, and FORCE_REFUND_GRACE has elapsed past endTime.
        //      This prevents contributors from being permanently locked
        //      out when the creator vanishes after a successful raise.
        require(
            presaleInfo.cancelled ||
            (block.timestamp > presaleInfo.endTime && presaleInfo.totalRaised < presaleInfo.softCap) ||
            (
                !presaleInfo.finalized &&
                block.timestamp > presaleInfo.endTime + FORCE_REFUND_GRACE
            ),
            "Refunds not available"
        );

        uint256 contribution = contributions[msg.sender];
        require(contribution > 0, "Nothing to refund");

        contributions[msg.sender] = 0;
        (bool ok, ) = payable(msg.sender).call{value: contribution}("");
        require(ok, "Refund transfer failed");

        emit Refunded(msg.sender, contribution);
    }

    function claim() external nonReentrant {
        require(presaleInfo.finalized, "Not finalized");
        require(contributions[msg.sender] > 0, "No contribution");
        require(!hasClaimed[msg.sender], "Already claimed");

        hasClaimed[msg.sender] = true;

        uint256 tokenAmount = (contributions[msg.sender] * presaleInfo.presaleRate) / 1e18;

        IERC20(presaleInfo.token).safeTransfer(msg.sender, tokenAmount);

        emit Claimed(msg.sender, tokenAmount);
    }

    // Two-phase launch: flip public trading on the bound token AFTER
    // the per-presale claim window has elapsed.
    //
    //   - Creator may call after `tradingOpensAt` (i.e. finalize + claimWindow).
    //   - Anyone may call after the 24h PUBLIC_OPEN_FAILSAFE so a
    //     missing/buggy creator can't strand contributors with locked
    //     LP and a tradeless token forever.
    //
    // We `try`/`catch` the token call so a botched/legacy token can't
    // grief this contract — `tradingOpened` is only set true on a
    // successful flip, and the call can be retried.
    function openTrading() external nonReentrant {
        require(presaleInfo.finalized, "Not finalized");
        require(!tradingOpened, "Already opened");

        bool publicFailsafe = false;
        if (msg.sender != presaleInfo.creator) {
            require(
                block.timestamp >= tradingOpensAt + PUBLIC_OPEN_FAILSAFE,
                "Not creator and failsafe not reached"
            );
            publicFailsafe = true;
        } else {
            require(block.timestamp >= tradingOpensAt, "Claim window still open");
        }

        // Idempotent fast-path: if the token's trading is already on
        // (e.g. owner manually flipped it), just record state and emit.
        try IHaloToken(presaleInfo.token).tradingEnabled() returns (bool already) {
            if (already) {
                tradingOpened = true;
                emit TradingOpened(msg.sender, publicFailsafe);
                return;
            }
        } catch {
            // ignore — try the real call below
        }

        // Spec requires the wrapper to be unstickable: even if the
        // token's enableTrading() reverts (legacy/non-Halo token, the
        // wrapper isn't bound, etc.), we MUST still mark tradingOpened
        // = true so the failsafe path can never leave contributors
        // permanently locked out of the "trading is open" UI / event
        // surface. Operator/contributors can then unstick the token
        // through other means (or accept it'll never trade) without
        // this contract being a blocker.
        try IHaloToken(presaleInfo.token).enableTrading() {
            tradingOpened = true;
            emit TradingOpened(msg.sender, publicFailsafe);
        } catch {
            tradingOpened = true;
            emit TradingOpened(msg.sender, publicFailsafe);
        }
    }

    function getPresaleStatus() external view returns (string memory) {
        if (presaleInfo.cancelled) return "Cancelled";
        if (presaleInfo.finalized) return "Finalized";
        if (block.timestamp < presaleInfo.startTime) return "Upcoming";
        if (block.timestamp < presaleInfo.endTime) return "Live";
        if (presaleInfo.totalRaised >= presaleInfo.softCap) return "Ended - Success";
        return "Ended - Failed";
    }

    // Two-phase launch view: returns the snapshot the UI needs to render
    // the countdown / "Open Trading" button without reading 4 separate
    // public state vars.
    function getClaimWindowState() external view returns (
        uint256 opensAt,
        uint256 windowSeconds,
        bool opened,
        uint256 publicFailsafeAt
    ) {
        return (
            tradingOpensAt,
            claimWindow,
            tradingOpened,
            tradingOpensAt == 0 ? 0 : tradingOpensAt + PUBLIC_OPEN_FAILSAFE
        );
    }

    function getUserContribution(address user) external view returns (uint256) {
        return contributions[user];
    }

    function getTimeRemaining() external view returns (uint256) {
        if (block.timestamp >= presaleInfo.endTime) return 0;
        if (block.timestamp < presaleInfo.startTime) {
            return presaleInfo.endTime - presaleInfo.startTime;
        }
        return presaleInfo.endTime - block.timestamp;
    }

    function getProgressPercent() external view returns (uint256) {
        if (presaleInfo.hardCap == 0) return 0;
        return (presaleInfo.totalRaised * 100) / presaleInfo.hardCap;
    }

    /**
     * @dev Burn unused tokens held by this wrapper AND any creator-residual
     * the wrapper has been approved for. Extracted from finalize() to keep
     * that function under the via-IR stack-too-deep limit.
     *
     * - `totalUnused` is the sum of unsold-presale + unused-liquidity tokens
     *   that are still sitting on the wrapper after addLiquidityETH; they
     *   are transferred straight to BURN_ADDRESS.
     * - The creator-residual portion uses transferFrom with the
     *   creator's allowance, bounded by their current balance, so we
     *   only ever pull what the creator has explicitly approved. The
     *   wizard pre-approves the wrapper for the full creator balance
     *   precisely so this path can clean up the ~80% residual.
     */
    function _burnExcess(uint256 totalUnused) internal {
        IERC20 token = IERC20(presaleInfo.token);
        if (totalUnused > 0) {
            token.safeTransfer(BURN_ADDRESS, totalUnused);
        }
        address creator = presaleInfo.creator;
        uint256 allowance_ = token.allowance(creator, address(this));
        uint256 balance_ = token.balanceOf(creator);
        uint256 residual = allowance_ < balance_ ? allowance_ : balance_;
        if (residual > 0) {
            // Two-hop via this wrapper instead of a direct creator→burn
            // transferFrom. The wrapper is exempt from the token's
            // fee/transfer/front-run guards (set via setPresaleContract);
            // a direct creator→0xdEaD path is NOT exempt and trips the
            // BEP20Token_Optimized E38 front-running guard whenever the
            // creator interacted with the token recently (which is
            // always the case during finalize since createPresale's
            // transferFrom set their `_lastBlockInteraction`).
            token.safeTransferFrom(creator, address(this), residual);
            token.safeTransfer(BURN_ADDRESS, residual);
        }
        emit ExcessBurned(totalUnused, residual);
    }

    function getPresaleInfo() external view returns (PresaleInfo memory) {
        return presaleInfo;
    }
}

contract HaloPresaleFactory {
    using SafeERC20 for IERC20;

    address public immutable presaleImplementation;
    address public immutable platformWallet;
    address public immutable router;
    address public immutable lpLocker;
    uint8 public immutable lockerKind; // 0=RAW_TRANSFER, 1=PINKLOCK_V2, 2=HALO_TIMELOCK_V2

    uint256 public presaleCount;

    // Per-creator presale counter. Used as the CREATE2 salt seed so the
    // wrapper address is deterministic from JUST (factory, creator,
    // creatorPresaleCount) — letting the wizard pre-compute the wrapper
    // address client-side and call `token.setPresaleContract(predicted)`
    // BEFORE invoking `createPresale(...)`. Race-free: each creator gets
    // its own monotonic counter so concurrent creators can't collide.
    mapping(address => uint256) public creatorPresaleCount;

    mapping(uint256 => address) public presales;
    mapping(address => bool) public isPresale;
    mapping(address => uint256[]) public userPresales;

    event PresaleCreated(
        uint256 indexed presaleId,
        address indexed presaleAddress,
        address indexed creator,
        address token
    );

    constructor(
        address _implementation,
        address _platformWallet,
        address _router,
        address _lpLocker,
        uint8 _lockerKind
    ) {
        require(_lockerKind <= 2, "Bad locker kind");
        presaleImplementation = _implementation;
        platformWallet = _platformWallet;
        router = _router;
        lpLocker = _lpLocker;
        lockerKind = _lockerKind;
    }

    function createPresale(
        address _token,
        uint256[11] memory _params,
        string memory _metadataURI
    ) external returns (address presaleAddress) {
        require(_token != address(0), "Invalid token");

        uint256 _tokensForPresale = (_params[0] * _params[2]) / 1e18;
        uint256 _tokensForLiquidity = (_params[0] * _params[6] * _params[3]) / (1e18 * 10000);
        uint256 totalRequired = _tokensForPresale + _tokensForLiquidity;

        require(
            IERC20(_token).balanceOf(msg.sender) >= totalRequired,
            "Insufficient token balance"
        );
        require(
            IERC20(_token).allowance(msg.sender, address(this)) >= totalRequired,
            "Insufficient factory allowance"
        );

        bytes memory bytecode = getCloneBytecode(presaleImplementation);
        // Deterministic salt: (creator, creatorPresaleCount[creator]).
        // NO block.timestamp — the wizard MUST be able to predict this
        // address client-side to call setPresaleContract before
        // createPresale.
        bytes32 salt = keccak256(abi.encodePacked(msg.sender, creatorPresaleCount[msg.sender]));

        assembly {
            presaleAddress := create2(0, add(bytecode, 0x20), mload(bytecode), salt)
        }

        // Token must report the EXACT platform fee wallet this factory
        // was wired to at deploy time. Closes the "rogue/forked token
        // routes platform fees to attacker wallet" griefing vector and
        // makes the platform-fee wallet enforcement on-chain provable
        // for scanners. We `try`/`catch` so a token that doesn't
        // implement the view at all surfaces a clean error instead of
        // a raw revert.
        try IHaloToken(_token).platformFeeWallet() returns (address w) {
            require(w == platformWallet, "Token platform wallet mismatch");
        } catch {
            revert("Token missing platformFeeWallet()");
        }

        IERC20(_token).safeTransferFrom(msg.sender, presaleAddress, totalRequired);

        HaloPresale(presaleAddress).initialize(
            msg.sender,
            _token,
            _params,
            _metadataURI,
            platformWallet,
            router,
            lpLocker,
            lockerKind
        );

        presaleCount++;
        creatorPresaleCount[msg.sender]++;
        presales[presaleCount] = presaleAddress;
        isPresale[presaleAddress] = true;
        userPresales[msg.sender].push(presaleCount);

        emit PresaleCreated(presaleCount, presaleAddress, msg.sender, _token);
    }

    function getCloneBytecode(address target) public pure returns (bytes memory) {
        bytes memory bytecode = new bytes(0x37);
        assembly {
            mstore(add(bytecode, 0x20), 0x3d602d80600a3d3981f3363d3d373d3d3d363d73000000000000000000000000)
            mstore(add(bytecode, 0x34), shl(0x60, target))
            mstore(add(bytecode, 0x48), 0x5af43d82803e903d91602b57fd5bf30000000000000000000000000000000000)
        }
        return bytecode;
    }

    // View helper for the wizard: returns the CREATE2 address the next
    // call to createPresale(...) from `creator` will deploy the wrapper
    // at. Mirrors the salt + initcode logic above so a single Solidity
    // change automatically updates the prediction.
    function predictNextPresaleAddress(address creator) external view returns (address predicted) {
        bytes memory initCode = getCloneBytecode(presaleImplementation);
        bytes32 salt = keccak256(abi.encodePacked(creator, creatorPresaleCount[creator]));
        bytes32 initCodeHash = keccak256(initCode);
        bytes32 raw = keccak256(
            abi.encodePacked(bytes1(0xff), address(this), salt, initCodeHash)
        );
        predicted = address(uint160(uint256(raw)));
    }

    function getUserPresaleCount(address user) external view returns (uint256) {
        return userPresales[user].length;
    }

    function getUserPresaleIds(address user) external view returns (uint256[] memory) {
        return userPresales[user];
    }

    function getPresaleAddress(uint256 presaleId) external view returns (address) {
        return presales[presaleId];
    }
}


// ===== FILE: _openzeppelin/contracts/utils/ReentrancyGuard.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/ReentrancyGuard.sol)

pragma solidity ^0.8.20;

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
 * TIP: If EIP-1153 (transient storage) is available on the chain you're deploying at,
 * consider using {ReentrancyGuardTransient} instead.
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
    uint256 private constant NOT_ENTERED = 1;
    uint256 private constant ENTERED = 2;

    uint256 private _status;

    /**
     * @dev Unauthorized reentrant call.
     */
    error ReentrancyGuardReentrantCall();

    constructor() {
        _status = NOT_ENTERED;
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
        // On the first call to nonReentrant, _status will be NOT_ENTERED
        if (_status == ENTERED) {
            revert ReentrancyGuardReentrantCall();
        }

        // Any calls to nonReentrant after this point will fail
        _status = ENTERED;
    }

    function _nonReentrantAfter() private {
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _status = NOT_ENTERED;
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        return _status == ENTERED;
    }
}


// ===== FILE: _openzeppelin/contracts/token/ERC20/IERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (token/ERC20/IERC20.sol)

pragma solidity >=0.4.16;

/**
 * @dev Interface of the ERC-20 standard as defined in the ERC.
 */
interface IERC20 {
    /**
     * @dev Emitted when `value` tokens are moved from one account (`from`) to
     * another (`to`).
     *
     * Note that `value` may be zero.
     */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /**
     * @dev Emitted when the allowance of a `spender` for an `owner` is set by
     * a call to {approve}. `value` is the new allowance.
     */
    event Approval(address indexed owner, address indexed spender, uint256 value);

    /**
     * @dev Returns the value of tokens in existence.
     */
    function totalSupply() external view returns (uint256);

    /**
     * @dev Returns the value of tokens owned by `account`.
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Moves a `value` amount of tokens from the caller's account to `to`.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transfer(address to, uint256 value) external returns (bool);

    /**
     * @dev Returns the remaining number of tokens that `spender` will be
     * allowed to spend on behalf of `owner` through {transferFrom}. This is
     * zero by default.
     *
     * This value changes when {approve} or {transferFrom} are called.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Sets a `value` amount of tokens as the allowance of `spender` over the
     * caller's tokens.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * IMPORTANT: Beware that changing an allowance with this method brings the risk
     * that someone may use both the old and the new allowance by unfortunate
     * transaction ordering. One possible solution to mitigate this race
     * condition is to first reduce the spender's allowance to 0 and set the
     * desired value afterwards:
     * https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
     *
     * Emits an {Approval} event.
     */
    function approve(address spender, uint256 value) external returns (bool);

    /**
     * @dev Moves a `value` amount of tokens from `from` to `to` using the
     * allowance mechanism. `value` is then deducted from the caller's
     * allowance.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}


// ===== FILE: _openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.3.0) (token/ERC20/utils/SafeERC20.sol)

pragma solidity ^0.8.20;

import {IERC20} from "../IERC20.sol";
import {IERC1363} from "../../../interfaces/IERC1363.sol";

/**
 * @title SafeERC20
 * @dev Wrappers around ERC-20 operations that throw on failure (when the token
 * contract returns false). Tokens that return no value (and instead revert or
 * throw on failure) are also supported, non-reverting calls are assumed to be
 * successful.
 * To use this library you can add a `using SafeERC20 for IERC20;` statement to your contract,
 * which allows you to call the safe operations as `token.safeTransfer(...)`, etc.
 */
library SafeERC20 {
    /**
     * @dev An operation with an ERC-20 token failed.
     */
    error SafeERC20FailedOperation(address token);

    /**
     * @dev Indicates a failed `decreaseAllowance` request.
     */
    error SafeERC20FailedDecreaseAllowance(address spender, uint256 currentAllowance, uint256 requestedDecrease);

    /**
     * @dev Transfer `value` amount of `token` from the calling contract to `to`. If `token` returns no value,
     * non-reverting calls are assumed to be successful.
     */
    function safeTransfer(IERC20 token, address to, uint256 value) internal {
        _callOptionalReturn(token, abi.encodeCall(token.transfer, (to, value)));
    }

    /**
     * @dev Transfer `value` amount of `token` from `from` to `to`, spending the approval given by `from` to the
     * calling contract. If `token` returns no value, non-reverting calls are assumed to be successful.
     */
    function safeTransferFrom(IERC20 token, address from, address to, uint256 value) internal {
        _callOptionalReturn(token, abi.encodeCall(token.transferFrom, (from, to, value)));
    }

    /**
     * @dev Variant of {safeTransfer} that returns a bool instead of reverting if the operation is not successful.
     */
    function trySafeTransfer(IERC20 token, address to, uint256 value) internal returns (bool) {
        return _callOptionalReturnBool(token, abi.encodeCall(token.transfer, (to, value)));
    }

    /**
     * @dev Variant of {safeTransferFrom} that returns a bool instead of reverting if the operation is not successful.
     */
    function trySafeTransferFrom(IERC20 token, address from, address to, uint256 value) internal returns (bool) {
        return _callOptionalReturnBool(token, abi.encodeCall(token.transferFrom, (from, to, value)));
    }

    /**
     * @dev Increase the calling contract's allowance toward `spender` by `value`. If `token` returns no value,
     * non-reverting calls are assumed to be successful.
     *
     * IMPORTANT: If the token implements ERC-7674 (ERC-20 with temporary allowance), and if the "client"
     * smart contract uses ERC-7674 to set temporary allowances, then the "client" smart contract should avoid using
     * this function. Performing a {safeIncreaseAllowance} or {safeDecreaseAllowance} operation on a token contract
     * that has a non-zero temporary allowance (for that particular owner-spender) will result in unexpected behavior.
     */
    function safeIncreaseAllowance(IERC20 token, address spender, uint256 value) internal {
        uint256 oldAllowance = token.allowance(address(this), spender);
        forceApprove(token, spender, oldAllowance + value);
    }

    /**
     * @dev Decrease the calling contract's allowance toward `spender` by `requestedDecrease`. If `token` returns no
     * value, non-reverting calls are assumed to be successful.
     *
     * IMPORTANT: If the token implements ERC-7674 (ERC-20 with temporary allowance), and if the "client"
     * smart contract uses ERC-7674 to set temporary allowances, then the "client" smart contract should avoid using
     * this function. Performing a {safeIncreaseAllowance} or {safeDecreaseAllowance} operation on a token contract
     * that has a non-zero temporary allowance (for that particular owner-spender) will result in unexpected behavior.
     */
    function safeDecreaseAllowance(IERC20 token, address spender, uint256 requestedDecrease) internal {
        unchecked {
            uint256 currentAllowance = token.allowance(address(this), spender);
            if (currentAllowance < requestedDecrease) {
                revert SafeERC20FailedDecreaseAllowance(spender, currentAllowance, requestedDecrease);
            }
            forceApprove(token, spender, currentAllowance - requestedDecrease);
        }
    }

    /**
     * @dev Set the calling contract's allowance toward `spender` to `value`. If `token` returns no value,
     * non-reverting calls are assumed to be successful. Meant to be used with tokens that require the approval
     * to be set to zero before setting it to a non-zero value, such as USDT.
     *
     * NOTE: If the token implements ERC-7674, this function will not modify any temporary allowance. This function
     * only sets the "standard" allowance. Any temporary allowance will remain active, in addition to the value being
     * set here.
     */
    function forceApprove(IERC20 token, address spender, uint256 value) internal {
        bytes memory approvalCall = abi.encodeCall(token.approve, (spender, value));

        if (!_callOptionalReturnBool(token, approvalCall)) {
            _callOptionalReturn(token, abi.encodeCall(token.approve, (spender, 0)));
            _callOptionalReturn(token, approvalCall);
        }
    }

    /**
     * @dev Performs an {ERC1363} transferAndCall, with a fallback to the simple {ERC20} transfer if the target has no
     * code. This can be used to implement an {ERC721}-like safe transfer that rely on {ERC1363} checks when
     * targeting contracts.
     *
     * Reverts if the returned value is other than `true`.
     */
    function transferAndCallRelaxed(IERC1363 token, address to, uint256 value, bytes memory data) internal {
        if (to.code.length == 0) {
            safeTransfer(token, to, value);
        } else if (!token.transferAndCall(to, value, data)) {
            revert SafeERC20FailedOperation(address(token));
        }
    }

    /**
     * @dev Performs an {ERC1363} transferFromAndCall, with a fallback to the simple {ERC20} transferFrom if the target
     * has no code. This can be used to implement an {ERC721}-like safe transfer that rely on {ERC1363} checks when
     * targeting contracts.
     *
     * Reverts if the returned value is other than `true`.
     */
    function transferFromAndCallRelaxed(
        IERC1363 token,
        address from,
        address to,
        uint256 value,
        bytes memory data
    ) internal {
        if (to.code.length == 0) {
            safeTransferFrom(token, from, to, value);
        } else if (!token.transferFromAndCall(from, to, value, data)) {
            revert SafeERC20FailedOperation(address(token));
        }
    }

    /**
     * @dev Performs an {ERC1363} approveAndCall, with a fallback to the simple {ERC20} approve if the target has no
     * code. This can be used to implement an {ERC721}-like safe transfer that rely on {ERC1363} checks when
     * targeting contracts.
     *
     * NOTE: When the recipient address (`to`) has no code (i.e. is an EOA), this function behaves as {forceApprove}.
     * Opposedly, when the recipient address (`to`) has code, this function only attempts to call {ERC1363-approveAndCall}
     * once without retrying, and relies on the returned value to be true.
     *
     * Reverts if the returned value is other than `true`.
     */
    function approveAndCallRelaxed(IERC1363 token, address to, uint256 value, bytes memory data) internal {
        if (to.code.length == 0) {
            forceApprove(token, to, value);
        } else if (!token.approveAndCall(to, value, data)) {
            revert SafeERC20FailedOperation(address(token));
        }
    }

    /**
     * @dev Imitates a Solidity high-level call (i.e. a regular function call to a contract), relaxing the requirement
     * on the return value: the return value is optional (but if data is returned, it must not be false).
     * @param token The token targeted by the call.
     * @param data The call data (encoded using abi.encode or one of its variants).
     *
     * This is a variant of {_callOptionalReturnBool} that reverts if call fails to meet the requirements.
     */
    function _callOptionalReturn(IERC20 token, bytes memory data) private {
        uint256 returnSize;
        uint256 returnValue;
        assembly ("memory-safe") {
            let success := call(gas(), token, 0, add(data, 0x20), mload(data), 0, 0x20)
            // bubble errors
            if iszero(success) {
                let ptr := mload(0x40)
                returndatacopy(ptr, 0, returndatasize())
                revert(ptr, returndatasize())
            }
            returnSize := returndatasize()
            returnValue := mload(0)
        }

        if (returnSize == 0 ? address(token).code.length == 0 : returnValue != 1) {
            revert SafeERC20FailedOperation(address(token));
        }
    }

    /**
     * @dev Imitates a Solidity high-level call (i.e. a regular function call to a contract), relaxing the requirement
     * on the return value: the return value is optional (but if data is returned, it must not be false).
     * @param token The token targeted by the call.
     * @param data The call data (encoded using abi.encode or one of its variants).
     *
     * This is a variant of {_callOptionalReturn} that silently catches all reverts and returns a bool instead.
     */
    function _callOptionalReturnBool(IERC20 token, bytes memory data) private returns (bool) {
        bool success;
        uint256 returnSize;
        uint256 returnValue;
        assembly ("memory-safe") {
            success := call(gas(), token, 0, add(data, 0x20), mload(data), 0, 0x20)
            returnSize := returndatasize()
            returnValue := mload(0)
        }
        return success && (returnSize == 0 ? address(token).code.length > 0 : returnValue == 1);
    }
}


// ===== FILE: _openzeppelin/contracts/interfaces/IERC1363.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (interfaces/IERC1363.sol)

pragma solidity >=0.6.2;

import {IERC20} from "./IERC20.sol";
import {IERC165} from "./IERC165.sol";

/**
 * @title IERC1363
 * @dev Interface of the ERC-1363 standard as defined in the https://eips.ethereum.org/EIPS/eip-1363[ERC-1363].
 *
 * Defines an extension interface for ERC-20 tokens that supports executing code on a recipient contract
 * after `transfer` or `transferFrom`, or code on a spender contract after `approve`, in a single transaction.
 */
interface IERC1363 is IERC20, IERC165 {
    /*
     * Note: the ERC-165 identifier for this interface is 0xb0202a11.
     * 0xb0202a11 ===
     *   bytes4(keccak256('transferAndCall(address,uint256)')) ^
     *   bytes4(keccak256('transferAndCall(address,uint256,bytes)')) ^
     *   bytes4(keccak256('transferFromAndCall(address,address,uint256)')) ^
     *   bytes4(keccak256('transferFromAndCall(address,address,uint256,bytes)')) ^
     *   bytes4(keccak256('approveAndCall(address,uint256)')) ^
     *   bytes4(keccak256('approveAndCall(address,uint256,bytes)'))
     */

    /**
     * @dev Moves a `value` amount of tokens from the caller's account to `to`
     * and then calls {IERC1363Receiver-onTransferReceived} on `to`.
     * @param to The address which you want to transfer to.
     * @param value The amount of tokens to be transferred.
     * @return A boolean value indicating whether the operation succeeded unless throwing.
     */
    function transferAndCall(address to, uint256 value) external returns (bool);

    /**
     * @dev Moves a `value` amount of tokens from the caller's account to `to`
     * and then calls {IERC1363Receiver-onTransferReceived} on `to`.
     * @param to The address which you want to transfer to.
     * @param value The amount of tokens to be transferred.
     * @param data Additional data with no specified format, sent in call to `to`.
     * @return A boolean value indicating whether the operation succeeded unless throwing.
     */
    function transferAndCall(address to, uint256 value, bytes calldata data) external returns (bool);

    /**
     * @dev Moves a `value` amount of tokens from `from` to `to` using the allowance mechanism
     * and then calls {IERC1363Receiver-onTransferReceived} on `to`.
     * @param from The address which you want to send tokens from.
     * @param to The address which you want to transfer to.
     * @param value The amount of tokens to be transferred.
     * @return A boolean value indicating whether the operation succeeded unless throwing.
     */
    function transferFromAndCall(address from, address to, uint256 value) external returns (bool);

    /**
     * @dev Moves a `value` amount of tokens from `from` to `to` using the allowance mechanism
     * and then calls {IERC1363Receiver-onTransferReceived} on `to`.
     * @param from The address which you want to send tokens from.
     * @param to The address which you want to transfer to.
     * @param value The amount of tokens to be transferred.
     * @param data Additional data with no specified format, sent in call to `to`.
     * @return A boolean value indicating whether the operation succeeded unless throwing.
     */
    function transferFromAndCall(address from, address to, uint256 value, bytes calldata data) external returns (bool);

    /**
     * @dev Sets a `value` amount of tokens as the allowance of `spender` over the
     * caller's tokens and then calls {IERC1363Spender-onApprovalReceived} on `spender`.
     * @param spender The address which will spend the funds.
     * @param value The amount of tokens to be spent.
     * @return A boolean value indicating whether the operation succeeded unless throwing.
     */
    function approveAndCall(address spender, uint256 value) external returns (bool);

    /**
     * @dev Sets a `value` amount of tokens as the allowance of `spender` over the
     * caller's tokens and then calls {IERC1363Spender-onApprovalReceived} on `spender`.
     * @param spender The address which will spend the funds.
     * @param value The amount of tokens to be spent.
     * @param data Additional data with no specified format, sent in call to `spender`.
     * @return A boolean value indicating whether the operation succeeded unless throwing.
     */
    function approveAndCall(address spender, uint256 value, bytes calldata data) external returns (bool);
}


// ===== FILE: _openzeppelin/contracts/interfaces/IERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (interfaces/IERC20.sol)

pragma solidity >=0.4.16;

import {IERC20} from "../token/ERC20/IERC20.sol";


// ===== FILE: _openzeppelin/contracts/interfaces/IERC165.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (interfaces/IERC165.sol)

pragma solidity >=0.4.16;

import {IERC165} from "../utils/introspection/IERC165.sol";


// ===== FILE: _openzeppelin/contracts/utils/introspection/IERC165.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (utils/introspection/IERC165.sol)

pragma solidity >=0.4.16;

/**
 * @dev Interface of the ERC-165 standard, as defined in the
 * https://eips.ethereum.org/EIPS/eip-165[ERC].
 *
 * Implementers can declare support of contract interfaces, which can then be
 * queried by others ({ERC165Checker}).
 *
 * For an implementation, see {ERC165}.
 */
interface IERC165 {
    /**
     * @dev Returns true if this contract implements the interface defined by
     * `interfaceId`. See the corresponding
     * https://eips.ethereum.org/EIPS/eip-165#how-interfaces-are-identified[ERC section]
     * to learn more about how these ids are created.
     *
     * This function call must use less than 30 000 gas.
     */
    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}
