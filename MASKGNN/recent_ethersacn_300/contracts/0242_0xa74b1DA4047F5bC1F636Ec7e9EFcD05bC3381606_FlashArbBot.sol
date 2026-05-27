// ===== FILE: FlashArbBot.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// ─────────────────────────────────────────────────────────────────────────────
//  FlashArbBot — Arbitrum One
//
//  Схема:
//    1. execute() — задаёшь токен, сумму займа, маршрут DEX
//    2. Aave V3 Arbitrum выдаёт flash loan
//    3. executeOperation() — свапы по маршруту
//    4. Возврат займа + 0.09% Aave fee
//    5. Прибыль остаётся на контракте → owner забирает через withdraw()
//
//  Деплой: Arbitrum One
//  Aave V3 Pool Arbitrum: 0x794a61358D6845594F94dc1DB02A252b5b4814aD
//
//  ДЕПЛОЙ ЧЕРЕЗ REMIX:
//    1. remix.ethereum.org
//    2. Сеть: Arbitrum One (chain_id 42161)
//    3. Constructor: _aavePool = 0x794a61358D6845594F94dc1DB02A252b5b4814aD
// ─────────────────────────────────────────────────────────────────────────────

// ── Interfaces ───────────────────────────────────────────────────────────────

interface IERC20 {
    function approve(address spender, uint256 amount) external returns (bool);
    function transfer(address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

// Aave V3 Pool
interface IPool {
    function flashLoanSimple(
        address receiverAddress,
        address asset,
        uint256 amount,
        bytes calldata params,
        uint16 referralCode
    ) external;
}

// Uniswap V2-style (Camelot V2, SushiSwap V2 Arbitrum)
interface IUniswapV2Router {
    function swapExactTokensForTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external returns (uint256[] memory amounts);
}

// Uniswap V3 SwapRouter (работает и на Arbitrum)
interface ISwapRouterV3 {
    struct ExactInputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24  fee;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
        uint160 sqrtPriceLimitX96;
    }
    function exactInputSingle(ExactInputSingleParams calldata params)
        external returns (uint256 amountOut);
}

// ── Structs ──────────────────────────────────────────────────────────────────

enum DexType {
    UniswapV2,  // 0 — Camelot V2, SushiSwap V2
    UniswapV3,  // 1 — Uniswap V3
    Sushiswap   // 2 — SushiSwap V2 (тот же интерфейс что V2)
}

struct SwapStep {
    DexType dex;
    address router;
    address tokenIn;
    address tokenOut;
    uint24  fee;           // для V3: 100/500/3000/10000; для V2: 0
    uint256 minAmountOut;  // ОБЯЗАТЕЛЬНО реальное значение из QuoterV2
}

// ── Contract ─────────────────────────────────────────────────────────────────

contract FlashArbBot {

    address public immutable owner;
    address public immutable aavePool;

    // Arbitrum Aave V3 Pool: 0x794a61358D6845594F94dc1DB02A252b5b4814aD

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyPool() {
        require(msg.sender == aavePool, "Not Aave Pool");
        _;
    }

    constructor(address _aavePool) {
        require(_aavePool != address(0), "Zero aave pool");
        owner    = msg.sender;
        aavePool = _aavePool;
    }

    // ── ТОЧКА ВХОДА ──────────────────────────────────────────────────────────

    /**
     * @param asset   токен займа (USDC: 0xaf88d065e77c8cC2239327C5EDb3A432268e5831)
     * @param amount  размер займа в wei (USDC: 1000 USDC = 1_000_000_000 = 1000 * 1e6)
     * @param steps   маршрут: минимум 2 шага,
     *                steps[0].tokenIn  == asset,
     *                steps[last].tokenOut == asset
     *
     * Важно: minAmountOut в каждом шаге должен быть рассчитан реально
     * через QuoterV2, не единица! Иначе транзакция будет убыточной.
     */
    function execute(
        address asset,
        uint256 amount,
        SwapStep[] calldata steps
    ) external onlyOwner {
        require(steps.length >= 2,                         "Min 2 steps");
        require(steps[0].tokenIn == asset,                 "Step0: tokenIn must be asset");
        require(steps[steps.length - 1].tokenOut == asset, "LastStep: tokenOut must be asset");

        // Каждый minAmountOut должен быть > 0
        for (uint256 i = 0; i < steps.length; i++) {
            require(steps[i].minAmountOut > 0, "minAmountOut must be > 0");
        }

        bytes memory params = abi.encode(steps, msg.sender);
        IPool(aavePool).flashLoanSimple(address(this), asset, amount, params, 0);
    }

    // ── CALLBACK от Aave ─────────────────────────────────────────────────────

    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external onlyPool returns (bool) {
        require(initiator == address(this), "Invalid initiator");

        (SwapStep[] memory steps, ) = abi.decode(params, (SwapStep[], address));

        uint256 current = amount;
        for (uint256 i = 0; i < steps.length; i++) {
            current = _swap(steps[i], current);
        }

        uint256 totalDebt = amount + premium;
        // Если прибыли нет — revert. Газ потрачен, но займ не взят.
        require(current >= totalDebt, "Unprofitable after fees");

        IERC20(asset).approve(aavePool, totalDebt);

        uint256 profit = current - totalDebt;
        emit ArbExecuted(asset, amount, profit, block.timestamp);

        return true;
    }

    // ── СВАП ─────────────────────────────────────────────────────────────────

    function _swap(SwapStep memory step, uint256 amountIn) internal returns (uint256) {
        IERC20(step.tokenIn).approve(step.router, amountIn);

        if (step.dex == DexType.UniswapV3) {
            return ISwapRouterV3(step.router).exactInputSingle(
                ISwapRouterV3.ExactInputSingleParams({
                    tokenIn:           step.tokenIn,
                    tokenOut:          step.tokenOut,
                    fee:               step.fee,
                    recipient:         address(this),
                    deadline:          block.timestamp + 180,  // 3 мин на Arbitrum
                    amountIn:          amountIn,
                    amountOutMinimum:  step.minAmountOut,
                    sqrtPriceLimitX96: 0
                })
            );
        } else {
            // UniswapV2-style (Camelot V2 / SushiSwap V2)
            address[] memory path = new address[](2);
            path[0] = step.tokenIn;
            path[1] = step.tokenOut;

            uint256[] memory amounts = IUniswapV2Router(step.router).swapExactTokensForTokens(
                amountIn,
                step.minAmountOut,
                path,
                address(this),
                block.timestamp + 180
            );
            return amounts[amounts.length - 1];
        }
    }

    // ── ВЫВОД ПРИБЫЛИ ────────────────────────────────────────────────────────

    /// Вывести ERC-20 прибыль (USDC, WETH и т.д.)
    function withdraw(address token) external onlyOwner {
        uint256 bal = IERC20(token).balanceOf(address(this));
        require(bal > 0, "Nothing to withdraw");
        IERC20(token).transfer(owner, bal);
        emit Withdrawn(token, bal);
    }

    /// Вывести ETH если накопился (например от rebates)
    function withdrawETH() external onlyOwner {
        uint256 bal = address(this).balance;
        require(bal > 0, "No ETH");
        (bool ok, ) = payable(owner).call{value: bal}("");
require(ok, "ETH transfer failed");
    }

    receive() external payable {}

    // ── EVENTS ────────────────────────────────────────────────────────────────

    event ArbExecuted(
        address indexed asset,
        uint256 loanAmount,
        uint256 profit,
        uint256 timestamp
    );
    event Withdrawn(address indexed token, uint256 amount);
}
