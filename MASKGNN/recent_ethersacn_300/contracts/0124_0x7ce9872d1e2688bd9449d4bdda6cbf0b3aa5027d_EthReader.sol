// ===== FILE: src/ethereum/EthReader.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import  "../types/response.sol";
import  "../types/request.sol";
import {IBalancerV3Reader} from "../common/IBalancerV3Reader.sol";
import {IBalancerV1Reader} from "../common/IBalancerV1Reader.sol";
import {IExtraPollingReader} from "../common/IExtraPollingReader.sol";
import {Erc4626Reader} from "../common/Erc4626Reader.sol";
import {ICurveLlamaLendReader} from "../common/ICurveLlamaLendReader.sol";
import {IOriginArmReader} from "../common/IOriginArmReader.sol";
import {IObricV2Reader} from "../common/IObricV2Reader.sol";
import {IDodoV1Reader} from "../common/IDodoV1Reader.sol";
import {IDodoGspReader} from "../common/IDodoGspReader.sol";
import {DoDoV1Response, DoDoGspResponse} from "../types/response.sol";
import {IEkuboReader} from "../common/IEkuboReader.sol";

contract EthReader {
    address public immutable ethArchive = 0x9129c541c7f44931c088Bb01342890238ef31661;
    IBalancerV3Reader public constant balancerV3Reader = IBalancerV3Reader(0x9587BC1b021e8B7573dE4a51dE9A81Fbb334bb38);
    IBalancerV1Reader public constant balancerV1Reader = IBalancerV1Reader(0x2A7fDeaF6b01E6e9B6CDED7bBDCF77996008ceEA);
    IExtraPollingReader public constant extraPollingReader = IExtraPollingReader(0xEBcD721A3404CBd19C67236710028A74490C1fB1);
    Erc4626Reader public constant erc4626Reader = Erc4626Reader(0x39BF5Eb764ea179D6a0284D297a9019428FD541d);
    ICurveLlamaLendReader public constant curveLlamaLendReader = ICurveLlamaLendReader(0xdCd5443888f0cA76DEA9282C86Fbe97AAE6325A4);
    IOriginArmReader public constant originArmReader = IOriginArmReader(0xce2bAD12747Bec4cb3D6F2846ae2aBbf166d0d2E);
    IObricV2Reader public constant obricV2Reader = IObricV2Reader(0x7e0276195d8559Ad135867CA02BE60B00Ff47ED3);
    IDodoV1Reader public constant dodoV1Reader = IDodoV1Reader(0x0DC9A67ED334c312ad3BA0873D20c7a6347F212a);
    IDodoGspReader public constant dodoGspReader = IDodoGspReader(0xabC5d20D83879E1d77ef26628f93F7dfAA51A63B);
    IEkuboReader public constant ekuboReader = IEkuboReader(0xD97d6F98E032b7030E4dB4f6BDAD9f2F3eE805C1);

    function version() public pure returns (uint256) {
        return 21;
    }

    fallback() external payable {
        address _impl = ethArchive;
        assembly {
            calldatacopy(0, 0, calldatasize())
            let result := delegatecall(gas(), _impl, 0, calldatasize(), 0, 0)
            returndatacopy(0, 0, returndatasize())
            switch result case 0 { revert(0, returndatasize()) } default { return(0, returndatasize()) }
        }
    }

    function balancerV3PoolFeatures(BalancerV3BasePoolRequest calldata params) external view returns (BalancerV3PoolFeatures memory) {
        return balancerV3Reader.balancerV3PoolFeatures(params);
    }

    function balancerV3BasePoolInfoV2(BalancerV3BasePoolRequest calldata request) external view returns (BalancerV3BasePoolResponseV2 memory response) {
        return balancerV3Reader.balancerV3BasePoolInfoV2(request);
    }

    function batchBalancerV3Erc4626Balance(address vault, address[] calldata rateProviders) external view returns (BalancerV3Erc4626BalanceResponse memory response) {
        return balancerV3Reader.batchBalancerV3Erc4626Balance(vault, rateProviders);
    }

    function balancerV1PoolInfo(BalancerV1PoolRequest calldata params) external view returns (BalancerV1PoolResponse memory) {
        return balancerV1Reader.balancerV1PoolInfo(params);
    }

    function isBalancerV1Pool(address pool, address factory) external view returns (bool) {
        return balancerV1Reader.isBalancerV1Pool(pool, factory);
    }

    function batchERC4626ConvertToAssets(address[] calldata tokens) external view returns (ExtraPollingERC4626Response memory response) {
        return extraPollingReader.batchERC4626ConvertToAssets(tokens);
    }

    function batchRateProviderGetRate(address[] calldata rateProviders) external view returns (ExtraPollingRateProviderResponse memory response) {
        return extraPollingReader.batchRateProviderGetRate(rateProviders);
    }

    function batchChainlinkOracleData(address[] calldata oracles) external view returns (ExtraPollingChainlinkResponse memory response) {
        return extraPollingReader.batchChainlinkOracleData(oracles);
    }

    function batchOracleRate(OracleQuery[] calldata queries) external view returns (ExtraPollingOracleResponse memory result){
        return extraPollingReader.batchOracleRate(queries);
    }

    function batchRebasingBalances(RebasingQuery[] calldata queries) external view returns (ExtraPollingRebasingResponse memory result){
        return extraPollingReader.batchRebasingBalances(queries);
    }

    function erc4626PoolInfo(address vault, address handler) external view returns (Erc4626Response memory result){
        return erc4626Reader.erc4626PoolInfo(vault, handler);
    }


    function curveLlamaLendPoolInfo(address poolAddr, int256 maxBandWindow) external view returns (CurveLlamaLendResponse memory) {
        return curveLlamaLendReader.curveLlamaLendPoolInfo(poolAddr, maxBandWindow);
    }

    function curveLlamaLendFactoryAmms(address[] calldata factories) external view returns (address[] memory) {
        return curveLlamaLendReader.curveLlamaLendFactoryAmms(factories);
    }

    function batchLlamaOracleRate(address[] calldata pools) external view returns (ExtraPollingLlamaOracleResponse memory result){
        return extraPollingReader.batchLlamaOracleRate(pools);
    }

    function originArmPoolInfo(OriginArmRequest calldata params) external view returns (OriginArmResponse memory) {
        return originArmReader.originArmPoolInfo(params);
    }

    function obricV2PoolInfo(address poolAddr) external view returns (ObricV2Response memory) {
        return obricV2Reader.obricV2PoolInfo(poolAddr);
    }

    function dodoV1PoolInfo(address pool) external view returns (DoDoV1Response memory) {
        return dodoV1Reader.dodoV1PoolInfo(pool);
    }

    function dodoGspPoolInfo(address pool) external view returns (DoDoGspResponse memory) {
        return dodoGspReader.dodoGspPoolInfo(pool);
    }

    function ekuboPoolInfo(EkuboPoolInfoRequest calldata params) external view returns (EkuboPoolInfoResponse memory) {
        return ekuboReader.ekuboPoolInfo(params);
    }

    function ekuboPoolInfoV3(EkuboPoolInfoRequestV3 calldata params) external view returns (EkuboPoolInfoResponse memory) {
        return ekuboReader.ekuboPoolInfoV3(params);
    }
}


// ===== FILE: src/types/response.sol =====
pragma solidity ^0.8.13;

import {TokenInfo, PoolData, HooksConfig} from "../abis/balancer_v3_vault.sol";

struct UniswapV1Response {
    uint blockNumber;
    address token;
    address factory;
    uint8 tokenDecimals;
    string tokenName;
    string tokenSymbol;
    uint256 ethReserve;
    uint256 tokenReserve;
    uint24 fee;
}

struct UniswapV2LikeResponse {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint24 fee;
}

struct FraxSwapV2Response {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint256 fee;
    // TWAMM
    uint256 twammReserve0;
    uint256 twammReserve1;
    uint256 lastVirtualOrderTimestamp;
    uint256 token0SalesRate;
    uint256 token1SalesRate;
    uint256 orderTimeInterval;
    // 订单到期：从 lastVirtualOrderTimestamp 起未来 48 小时内各到期时间点的 sales rate ending
    uint256 salesRateEndingCount;
    uint256[] salesRateEndingTimestamps;
    uint256[] orderPool0SalesRateEnding;
    uint256[] orderPool1SalesRateEnding;
    bool newSwapsPaused ;
}

struct TickInfo {
    int24 tick;
    int128 liquidityNet;
}

struct EkuboTickInfo {
    int32 tick;
    int128 liquidityNet;
}

struct UniswapV3LikeResponse {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint24 fee;
    uint256 liquidity;
    uint256 sqrtPriceX96;
    int24 tick;
    uint24 tickIndex;
    TickInfo[] ticks;
    int24 tickSpacing;
}

struct UniswapV3LikeTicksResponse {
    uint blockNumber;
    TickInfo[] ticks;
}

struct UniswapV4LikeResponse {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint24 lpFee;
    int24 tickSpacing;
    address hooks;
    uint24 protocolFee;
    uint128 liquidity;
    uint160 sqrtPriceX96;
    int24 tick;
    uint24 tickIndex;
    TickInfo[] ticks;
}

struct UniswapV4LikeResponseV2 {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint24 lpFee;
    int24 tickSpacing;
    address hooks;
    uint24 protocolFee;
    uint128 liquidity;
    uint160 sqrtPriceX96;
    int24 tick;
    uint24 tickIndex;
    TickInfo[] ticks;
    uint24 fee;
}

struct PancakeInfClResponse {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint24 lpFee;
    int24 tickSpacing;
    address hooks;
    uint24 protocolFee;
    uint128 liquidity;
    uint160 sqrtPriceX96;
    int24 tick;
    uint24 tickIndex;
    TickInfo[] ticks;
    bytes32 parameters;
}

struct PancakeInfClResponseV2 {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint24 lpFee;
    int24 tickSpacing;
    address hooks;
    uint24 protocolFee;
    uint128 liquidity;
    uint160 sqrtPriceX96;
    int24 tick;
    uint24 tickIndex;
    TickInfo[] ticks;
    bytes32 parameters;
    uint24 fee;
}

struct UniswapV4LikeTicksResponse {
    uint blockNumber;
    TickInfo[] ticks;
}

struct PancakeStableResponse {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 balance0;
    uint256 balance1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint256 fee;
    uint256[2] rates;
    uint256 nCoins;
    uint256 futureATime;
    uint256 futureA;
    uint256 initialATime;
    uint256 initialA;
    uint256 precision;
    uint256 feeDenominator;
    bool isKilled;
}

struct BinInfo {
    uint24 binId;
    uint128 binReserveX;
    uint128 binReserveY;
}

struct PancakeInfBinResponse {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint24 lpFee;
    bytes32 parameters;
    uint16 binStep;
    address hooks;
    uint24 protocolFee;
    uint24 activeId;
    uint24 activeBinIndex;
    BinInfo[] bins;
}

struct PancakeInfBinResponseV2 {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint24 lpFee;
    bytes32 parameters;
    uint16 binStep;
    address hooks;
    uint24 protocolFee;
    uint24 activeId;
    uint24 activeBinIndex;
    BinInfo[] bins;
    uint24 fee;
}

struct PancakeInfBinBinsResponse {
    uint blockNumber;
    BinInfo[] bins;
}

struct DoDoV2ResponseV2 {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint24 lpFee;
    uint24 mtFee;
    uint256 i;
    uint256 K;
    uint256 B;
    uint256 Q;
    uint256 B0;
    uint256 Q0;
    uint256 R;
}

// DodoV1/GSP 复用与 DoDoV2ResponseV2 相同的字段布局，Go 侧 unpack 结构一致
struct DoDoV1Response {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint24 lpFee;
    uint24 mtFee;
    uint256 i;
    uint256 K;
    uint256 B;
    uint256 Q;
    uint256 B0;
    uint256 Q0;
    uint256 R;
}

struct DoDoGspResponse {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint24 lpFee;
    uint24 mtFee;
    uint256 i;
    uint256 K;
    uint256 B;
    uint256 Q;
    uint256 B0;
    uint256 Q0;
    uint256 R;
}

struct MaverickV2Tick {
    int32 tick;
    uint256 reserveA;
    uint256 reserveB;
}

struct MaverickV2Response {
    uint blockNumber;
    address tokenA;
    address tokenB;
    uint8 decimalsA;
    uint8 decimalsB;
    string nameA;
    string nameB;
    string symbolA;
    string symbolB;
    uint256 feeAIn;
    uint256 feeBIn;
    uint256 tickSpacing;
    int32 activeTick;
    MaverickV2Tick[] ticks;
    uint256 reserveA; // 池子内 tokenA 余额（与 UniswapV3Like reserve 一致）
    uint256 reserveB; // 池子内 tokenB 余额
}

/// @notice 扩展 tick 数据（仅 ticks，用于与主池合并）
struct MaverickV2ExtraTicksResponse {
    uint blockNumber;
    uint256 tickSpacing;
    MaverickV2Tick[] ticks;
}

struct WombatResponse {
    uint blockNumber;
    uint256 nCoins;
    address[] tokens;
    bool[] isPaused;
    uint8[] decimals;
    string[] names;
    string[] symbols;
    uint256[] cash;
    uint256[] liability;
    uint256[] relativePrices;
    uint256 ampFactor;
    uint256 haircutRate;
    uint128 startCovRatio;
    uint128 endCovRatio;
}

struct AlgebraV1LikeResponse {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint24 fee;
    uint256 liquidity;
    uint256 sqrtPriceX96;
    int24 tick;
    uint24 tickIndex;
    TickInfo[] ticks;

    uint16 feeConfigAlpha1; // max value of the first sigmoid
    uint16 feeConfigAlpha2; // max value of the second sigmoid
    uint32 feeConfigBeta1; // shift along the x-axis for the first sigmoid
    uint32 feeConfigBeta2; // shift along the x-axis for the second sigmoid
    uint16 feeConfigGamma1; // horizontal stretch factor for the first sigmoid
    uint16 feeConfigGamma2; // horizontal stretch factor for the second sigmoid
    uint32 feeConfigVolumeBeta; // shift along the x-axis for the outer volume-sigmoid
    uint16 feeConfigVolumeGamma; // horizontal stretch factor the outer volume-sigmoid
    uint16 feeConfigBaseFee; // minimum possible fee

    // 以下是新增字段
    uint16 timepointIndex;

    uint32[] timepointBlockTimestamps;
    uint112[] timepointVolatilityCumulatives;
    uint256[] timepointVolumePerAvgLiquiditys;
}

struct AlgebraV1LikeResponseV2 {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint24 feeZto;
    uint24 feeOtz;
    uint256 liquidity;
    uint256 sqrtPriceX96;
    int24 tick;
    uint24 tickIndex;
    TickInfo[] ticks;
}

struct AlgebraV1LikeTicksResponse {
    uint blockNumber;
    TickInfo[] ticks;
}

struct AlgebraIntegralBasePool {
    address token0;
    address token1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint8 decimals0;
    uint8 decimals1;
    uint256 reserve0;
    uint256 reserve1;
    uint24 fee;
    uint256 liquidity;
    uint256 sqrtPrice;
    address plugin;
    uint8 pluginConfig;
    int24 tick;
    uint24 tickIndex;
    TickInfo[] ticks;
}

struct DynamicFeeConfig {
    uint16 alpha1;
    uint16 alpha2;
    uint32 beta1;
    uint32 beta2;
    uint16 gamma1;
    uint16 gamma2;
    uint16 baseFee;
}

struct SlidingFeeConfig {
    uint128 zeroToOneFeeFactor;
    uint128 oneToZeroFeeFactor;
    uint16 priceChangeFactor;
    uint16 baseFee;
}

struct ThenaIntegralResponse {
    uint blockNumber;
    AlgebraIntegralBasePool basePoolInfo;
    bool pluginPaused;
    bool feeType;
    DynamicFeeConfig dynamicFeeConfig;
    SlidingFeeConfig slidingFeeConfig;
    bool discountIsActive;
    uint32[] timepointBlockTimestamps;
    int56[] timepointTickCumulatives;
    uint88[] timepointVolatilityCumulatives;
}

struct QuickswapV4Response {
    uint blockNumber;
    AlgebraIntegralBasePool basePoolInfo;
    DynamicFeeConfig dynamicFeeConfig;
    uint32[] timepointBlockTimestamps;
    int56[] timepointTickCumulatives;
    uint88[] timepointVolatilityCumulatives;
}

struct CypherV4Response {
    uint blockNumber;
    AlgebraIntegralBasePool basePoolInfo;
    bool dynamicFeeEnabled;
    bool slidingFeeEnabled;
    DynamicFeeConfig dynamicFeeConfig;
    SlidingFeeConfig slidingFeeConfig;
    uint16 timepointIndex;
    int24 lastTick;
    uint32[] timepointBlockTimestamps;
    int56[] timepointTickCumulatives;
    uint88[] timepointVolatilityCumulatives;
}

struct CamelotV4Response {
    uint blockNumber;
    AlgebraIntegralBasePool basePoolInfo;
    bool dynamicFeeEnabled;
    bool slidingFeeEnabled;
    DynamicFeeConfig dynamicFeeConfig;
    SlidingFeeConfig slidingFeeConfig;
    uint16 timepointIndex;
    int24 lastTick;
    uint32[] timepointBlockTimestamps;
    int56[] timepointTickCumulatives;
    uint88[] timepointVolatilityCumulatives;
}

struct WDexResponse {
    uint blockNumber;
    AlgebraIntegralBasePool basePoolInfo;
    bool dynamicFeeEnabled;
    DynamicFeeConfig dynamicFeeConfig;
    uint16 timepointIndex;
    int24 lastTick;
    uint32[] timepointBlockTimestamps;
    int56[] timepointTickCumulatives;
    uint88[] timepointVolatilityCumulatives;
    uint16 discountBps;
}

struct FluidDexResponse {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint256 collateralTokenInRealReserves;
    uint256 collateralTokenOutRealReserves;
    uint256 debtTokenInRealReserves;
    uint256 debtTokenOutRealReserves;
    uint256 collateralTokenInImaginaryReserves;
    uint256 collateralTokenOutImaginaryReserves;
    uint256 debtTokenInImaginaryReserves;
    uint256 debtTokenOutImaginaryReserves;
    uint256 lastStoredPrice;
    uint256 centerPrice;
    uint256 upperRange;
    uint256 lowerRange;
    uint256 geometricMean;
    uint256 supplyToken0ExchangePrice;
    uint256 borrowToken0ExchangePrice;
    uint256 supplyToken1ExchangePrice;
    uint256 borrowToken1ExchangePrice;
    uint256 fee;
    uint256 withdrawableToken0Available;
    uint256 withdrawableToken0ExpandsTo;
    uint256 withdrawableToken0ExpandDuration;
    uint256 withdrawableToken1Available;
    uint256 withdrawableToken1ExpandsTo;
    uint256 withdrawableToken1ExpandDuration;
    uint256 borrowableToken0Available;
    uint256 borrowableToken0ExpandsTo;
    uint256 borrowableToken0ExpandDuration;
    uint256 borrowableToken1Available;
    uint256 borrowableToken1ExpandsTo;
    uint256 borrowableToken1ExpandDuration;
    uint256 token0UtilizationLimit;
    uint256 token1UtilizationLimit;
    uint256 token0LiquidityLayerUtilization;
    uint256 token1LiquidityLayerUtilization;
    bool token0Paused;
    bool token1Paused;
    bool isSwapAndArbitragePaused;
    uint256 token0LiquidityBalance;
    uint256 token1LiquidityBalance;
    uint256 token0TotalAmounts;
    uint256 token0ExchangePricesAndConfig;
    uint256 token0Configs2;
    uint256 token1TotalAmounts;
    uint256 token1ExchangePricesAndConfig;
    uint256 token1Configs2;
    // token precision immutables from constantsView2 (used for GetCollateralReserves/GetDebtReserves)
    uint256 token0NumeratorPrecision;
    uint256 token0DenominatorPrecision;
    uint256 token1NumeratorPrecision;
    uint256 token1DenominatorPrecision;
    uint256 token0SupplyTokenData;
    uint256 token0BorrowTokenData;
    uint256 token1SupplyTokenData;
    uint256 token1BorrowTokenData;
}

struct PositionFeesInfo {
    uint256 amount0;
    uint256 amount1;
}

enum FeeInPreference {
    Both,
    Paired,
    Clanker
}

struct PoolDynamicConfigVars {
    uint24 baseFee;
    uint24 maxLpFee;
    uint256 referenceTickFilterPeriod;
    uint256 resetPeriod;
    int24 resetTickFilter;
    uint256 feeControlNumerator;
    uint24 decayFilterBps;
}

struct PoolDynamicFeeVars {
    int24 referenceTick;
    int24 resetTick;
    uint256 resetTickTimestamp;
    uint256 lastSwapTimestamp;
    uint24 appliedVR;
    uint24 prevVA;
}

struct ClankerHookV2Response {
    bool clankerIsToken0;
    uint256 poolCreationTimestamp;
    bool mevModuleEnabled;
    uint256 maxMevModuleDelay;
    address lockerAddress;
    bool hasLocker;
    uint256 positionId;
    uint256 numPositions;
    uint16[] rewardBps;
    FeeInPreference[] feePreferences;
    PositionFeesInfo[] positionFees;
    uint24 clankerFee;
    uint24 pairedFee;
    PoolDynamicConfigVars configVars;
    PoolDynamicFeeVars feeVars;
}

struct ClankerHookResponse {
    bool clankerIsToken0;
    uint256 poolCreationTimestamp;
    bool mevModuleEnabled;
    uint256 maxMevModuleDelay;
    address lockerAddress;
    bool hasLocker;
    uint256 positionId;
    uint256 numPositions;
    uint16[] rewardBps;
    FeeInPreference[] feePreferences;
    PositionFeesInfo[] positionFees;
    uint24 clankerFee;
    uint24 pairedFee;
    PoolDynamicConfigVars configVars;
    PoolDynamicFeeVars feeVars;
    address mevModuleAddress;
    uint256 mevNextAuctionBlock;  // V1/V2: nextAuctionBlock (auction 阶段)
    uint256 mevDecayStartTime;    // V2 only: poolDecayStartTime (0 表示未开始 decay)
    uint24 mevStartingFee;        // V2 only: feeConfig.startingFee
    uint24 mevEndingFee;          // V2 only: feeConfig.endingFee
    uint256 mevSecondsToDecay;    // V2 only: feeConfig.secondsToDecay
    uint256 mevRound;             // V2 only: 当前 auction 轮次
    uint256 mevBlocksBetweenAuction; // V2 only: 两轮 auction 之间的区块间距
    uint256 mevPoolUnlockTime;      // V2 only: pool unlock time
}

struct SlipstreamBasePool {
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint24 fee;
    uint128 liquidity;
    uint160 sqrtPrice;
    int24 tick;
    int24 tickSpacing;
    address factory;
    uint16 observationIndex;
    uint16 observationCardinality;
    uint24 tickIndex;
    TickInfo[] ticks;
    uint256 reserve0;  // 添加到末尾以保持向后兼容性
    uint256 reserve1;  // 添加到末尾以保持向后兼容性
}

struct SlipstreamResponse {
    uint blockNumber;
    SlipstreamBasePool basePoolInfo;
    bool hasDynamicFee;
    address feeModuleAddr;
    uint24 baseFee;
    uint24 feeCap;
    uint64 scalingFactor;
    uint256 defaultScalingFactor;
    uint256 defaultFeeCap;
    uint32 secondsAgo;
    uint32[] observationTimestamps;
    int56[] observationTickCumulatives;
}

struct SlipstreamFactoriesOutput {
    bool isPool;
    address factory;
}

struct CurveStableResponseV3 {
    uint256 nCoins;
    uint blockNumber;
    address[] tokens;
    uint256[] decimals;
    uint256[] balances;
    string[] names;
    string[] symbols;
    uint256 [] assetTypes;
    uint256 [] storeRates;
    uint256 fee;
    uint256 adminFee;
    uint256 offPegFeeMultiplier;
    uint256 futureATime;
    uint256 futureA;
    uint256 initialATime;
    uint256 initialA;
    uint256 aPrecision;
    bool isMeta;
    CurveMetaPool metaPool;
    uint256[] actualBalances;
}

struct CurveMetaPool{
    uint256 nCoins;
    uint256 a;
    uint256 fee;
    uint256[] balances;
    uint256[] decimals;
    address[] tokens;
    string[] names;
    string[] symbols;
    address basePoolAddr;
    address lpToken;
    uint256 totalSupply;
    bool ng;
    uint256 futureATime;
    uint256 futureA;
    uint256 initialATime;
    uint256 initialA;
    uint256[] StoredRates;
    uint256 OffPegFeeMultiplier;
}

struct CurveBaseInfoResponse{
    bool isRegistered;
    bool isMeta;
    uint256 nCoins;
    uint256 baseNCoins;
}

struct CurveFilterResponseV2{
    bool isRegistered;
    bool isMeta;
    bool isLeading;
    address factory;
    uint256 nCoins;
    uint256 baseNCoins;
    uint256 [] assetTypes;
    address oracle;
    uint256 [] storeRates;
}

struct CamelotV2Response {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint112 reserve0;
    uint112 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    bool stableSwap;
    uint precisionMultiplier0;
    uint precisionMultiplier1;
    uint16 token0FeePercent;
    uint16 token1FeePercent;
}


struct AlgebraV1FeeConfiguration {
    uint16 alpha1;
    uint16 alpha2;
    uint32 beta1;
    uint32 beta2;
    uint16 gamma1;
    uint16 gamma2;
    uint32 volumeBeta;
    uint16 volumeGamma;
    uint16 baseFee;
}

struct AlgebraV1DirFeeLikeResponse {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint24 feeZto;
    uint24 feeOtz;
    uint256 liquidity;
    uint256 sqrtPriceX96;
    int24 tick;
    uint24 tickIndex;
    TickInfo[] ticks;
    AlgebraV1FeeConfiguration feeConfigZto;
    AlgebraV1FeeConfiguration feeConfigOtz;
    uint16 timepointIndex;
    uint32[] timepointBlockTimestamps;
    uint112[] timepointVolatilityCumulatives;
    uint256[] timepointVolumePerAvgLiquiditys;
}

struct AlgebraV1DirFeeLikeResponseV2 {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint24 feeZto;
    uint24 feeOtz;
    uint256 liquidity;
    uint256 sqrtPriceX96;
    int24 tick;
    uint24 tickIndex;
    TickInfo[] ticks;
    AlgebraV1FeeConfiguration feeConfigZto;
    AlgebraV1FeeConfiguration feeConfigOtz;
    uint16 timepointIndex;
    uint32[] timepointBlockTimestamps;
    int56[] timepointTickCumulatives;
    uint112[] timepointVolatilityCumulatives;
    uint256[] timepointVolumePerAvgLiquiditys;
}

struct CurveCryptoResponseV2 {
    uint256 nCoins;                // 代币数量
    uint256 blockNumber;           // 区块号
    address[] tokens;              // 代币地址数组
    uint256[] decimals;            // 精度数组
    uint256[] balances;            // 余额数组
    string[] names;                // 名称数组
    string[] symbols;              // 符号数组
    uint256 futureAGamma;        //打包的[A,gamma]
    uint256 initialAGamma;       //打包的[A,gamma]
    uint256 futureAGammaTime;   //未来A和gamma的时间戳
    uint256 initialAGammaTime;  //未来A和gamma的时间戳
    uint256[] priceScales;         // 内部价格比例2币种1个，3币种2个
    uint256 D;                     //缓存的D
    uint256 feeGamma;             // 手续费gamma
    uint256 midFee;               // 中间手续费
    uint256 outFee;               // 外部手续费
    bool isKilled;
    uint256[] actualBalances;     // 真实余额数组
}

struct BalancerV2WeightedResponse {
    uint blockNumber;
    bytes32 poolId;
    address poolAddress;
    address vault;
    address[] tokens;
    uint256[] balances;
    uint256[] weights;
    uint8[] decimals;
    string[] names;
    string[] symbols;
    uint256 swapFeePercentage;
    bool paused;
}

struct BalancerV1PoolResponse {
    uint blockNumber;
    address poolAddress;
    address[] tokens;
    uint256[] balances;
    uint256[] denormWeights;
    uint256 totalWeight;
    uint256 swapFee;
    bool finalized;
    bool publicSwap;
    uint8[] decimals;
    string[] names;
    string[] symbols;
}

struct BalancerV2StableResponse {
    uint blockNumber;
    bytes32 poolId;
    address poolAddress;
    address vault;
    address[] tokens;
    uint256[] balances;
    uint8[] decimals;
    string[] names;
    string[] symbols;
    uint256 swapFeePercentage;
    uint256 amplificationParameter;  // 当前 Amp 值（实时计算，已包含线性插值）
    bool isAmpUpdating;              // Amp 是否正在调整中
    uint256 bptIndex;
    address[] rateProviders;         // Rate Provider 列表（地址为 0 表示无 rate provider）
    bool hasRateProvider;            // 是否有任何 rate provider（快速判断标志）
    bool paused;
}

struct BalancerV2PoolTypeResponse {
    bool isValid;                    // 是否是有效的 Balancer 池子
    uint8 poolType;                  // 池子类型编码
    string poolTypeName;             // 池子类型名称
}

struct BalancerV2EclpPoolTypeResponse {
    bool isValid;                    // 是否是有效的 ECLP 池子
}

struct BalancerV2EclpResponse {
    uint blockNumber;
    bytes32 poolId;
    address[] tokens;
    uint256[] balances;
    uint8[] decimals;
    string[] names;
    string[] symbols;
    uint256 swapFeePercentage;
    // ECLP 参数
    int256 alpha;
    int256 beta;
    int256 c;
    int256 s;
    int256 lambda;
    // ECLP 派生参数
    int256 tauAlphaX;
    int256 tauAlphaY;
    int256 tauBetaX;
    int256 tauBetaY;
    int256 u;
    int256 v;
    int256 w;
    int256 z;
    int256 dSq;
    bool paused;
}


enum BalancerV3PoolType {
    UNKNOWN,
    WEIGHTED,
    STABLE,
    GYRO_ECLP,
    RECLAMM,
    QUANTAMM_WEIGHTED
}

struct BalancerV3PoolFeatures {
    BalancerV3PoolType poolType;
    address hooksContract;
}

struct BalancerV3BasePoolResponse {
    uint blockNumber;
    address[] tokens;
    TokenInfo[] tokenInfo;
    uint8[] decimals;
    string[] names;
    string[] symbols;
    uint256[] balancesRaw;
    uint256[] tokenRates;
    uint256[] decimalScalingFactors;
    uint256 swapFeePercentage;
    address hooksContract;
    bool enableHookAdjustedAmounts;
    bool shouldCallComputeDynamicSwapFee;
    bool shouldCallBeforeSwap;
    bool shouldCallAfterSwap;
    bool isVaultPaused;
    bool isPoolPaused;
}

struct RateProviderInfo {
    uint8 tokenIndex;
    address erc4626;
    address underlyingAsset;
    uint8 erc4626Decimals;
    uint8 underlyingDecimals;
    string underlyingName;
    string underlyingSymbol;
    bool isValid;
    address vaultAssetFeed; // non-zero for composite RPs (e.g. AaveMarketRateTransformer)
}

struct BalancerV3BasePoolResponseV2 {
    uint blockNumber;
    address[] tokens;
    TokenInfo[] tokenInfo;
    uint8[] decimals;
    string[] names;
    string[] symbols;
    uint256[] balancesRaw;
    uint256[] tokenRates;
    uint256[] decimalScalingFactors;
    uint256 swapFeePercentage;
    address hooksContract;
    bool enableHookAdjustedAmounts;
    bool shouldCallComputeDynamicSwapFee;
    bool shouldCallBeforeSwap;
    bool shouldCallAfterSwap;
    bool isVaultPaused;
    bool isPoolPaused;
    RateProviderInfo[] rateProviders;
}

struct LidoResponse {
    uint blockNumber;
    uint256 TotalPooledEther;
    uint256 TotalShares;
    uint256 StakeLimit;
}


struct EkuboPoolInfoResponse {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint64 Fee;
    uint32 tickSpacing;
    address hooks;
    uint128 liquidity;
    uint96 sqrtPriceX96;
    int32 tick;
    uint256 tickIndex;
    EkuboTickInfo[] ticks;
    // TWAMM extension fields (non-zero only when hooks == TWAMM address)
    uint32 twammLastTime;
    uint112 twammSaleRateToken0;
    uint112 twammSaleRateToken1;
}

struct NftStrategyHookResponse {
    uint256 deploymentBlock;
}

struct BancorV2PoolInfoResponse {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address reserveToken0;
    address reserveToken1;
    uint256 reserveBalance0;
    uint256 reserveBalance1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint32 fee;
    address anchor;
}

struct BancorV2IsValidResponse {
    bool isValid; // 是否是有效的 Converter（50/50 权重）
}

struct RingSwapV2Response {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;              // fwToken0 地址
    address token1;              // fwToken1 地址
    uint256 reserve0;
    uint256 reserve1;
    string name0;                // fwToken0 的 name
    string name1;                // fwToken1 的 name
    string symbol0;              // fwToken0 的 symbol
    string symbol1;              // fwToken1 的 symbol
    address originalToken0;        // 原始 token0 地址（通过 fwToken0.token() 获取）
    address originalToken1;        // 原始 token1 地址（通过 fwToken1.token() 获取）
    uint8 originalDecimals0;
    uint8 originalDecimals1;
    string originalName0;
    string originalName1;
    string originalSymbol0;
    string originalSymbol1;
    uint256 originalToken0Balance;
    uint256 originalToken1Balance;
}

struct FwTokenInfo {
    address originalToken;      // 原始 token 地址
    uint8 originalDecimals;     // 原始 token 的 decimals
    string originalName;        // 原始 token 的 name
    string originalSymbol;      // 原始 token 的 symbol
    uint256 originalTokenBalance; // 原始 token 的余额
}

struct BancorV3PoolInfoResponse {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address reserveToken0;
    address reserveToken1;
    uint256 reserveBalance0;
    uint256 reserveBalance1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint32 fee;
    bool tradingEnabled;
}

struct BancorV3IsValidResponse {
    bool isValid;
}

struct LfjV2PoolInfoResponse {
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint16 binStep;
    uint24 activeId;
    uint24 activeBinIndex;
    BinInfo[] bins;
    bytes32 hooksParameters;
    address hooks;
    uint8 hooksType;
    uint16 baseFactor;
    uint16 filterPeriod;
    uint16 decayPeriod;
    uint16 reductionFactor;
    uint24 variableFeeControl;
    uint16 protocolShare;
    uint24 maxVolatilityAccumulated;
    uint24 volatilityAccumulated;
    uint24 volatilityReference;
    uint24 indexRef;
    uint40 time;
}

struct NomiswapStableResponse {
    uint blockNumber;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    uint32 swapFee;
    uint256 A;
    uint8 token0Decimals;
    uint8 token1Decimals;
    string token0Name;
    string token1Name;
    string token0Symbol;
    string token1Symbol;
}

struct FlaunchHookResponse {
    uint24 swapFee;
    uint256 claimableFeesAmount0;
    uint256 claimableFeesAmount1;
}

struct LitePsmResponse {
    uint256 blockNumber;
    address gem;
    address dai;
    uint8 decimalsGem;
    uint8 decimalsDai;
    uint256 gemReserve;
    uint256 daiReserve;
    string gemName;
    string daiName;
    string gemSymbol;
    string daiSymbol;
    address pocket;
    uint256 to18ConversionFactor;
    uint256 tin;
    uint256 tout;
    uint256 HALTED;
    bool buyGemHalted;
    bool sellGemHalted;
}

struct SparkPsmResponse {
    uint256 blockNumber;
    // token0 = USDC
    address usdc;
    uint8   usdcDecimals;
    string  usdcName;
    string  usdcSymbol;
    // token1 = USDS
    address usds;
    uint8   usdsDecimals;
    string  usdsName;
    string  usdsSymbol;
    // token2 = sUSDS (yield-bearing)
    address susds;
    uint8   susdsDecimals;
    string  susdsName;
    string  susdsSymbol;
    address pocket;
    uint256 usdcReserve;   // USDC balance of pocket
    uint256 usdsReserve;   // USDS balance of PSM contract
    uint256 susdsReserve;  // sUSDS balance of PSM contract
}

struct DaiUsdsResponse {
    uint256 blockNumber;
    address dai;
    address usds;
    uint8 decimalsDai;
    uint8 decimalsUsds;
    uint256 daiReserve;
    uint256 usdsReserve;
    string daiName;
    string usdsName;
    string daiSymbol;
    string usdsSymbol;
}

struct UsddPsmResponse {
    uint256 blockNumber;
    address gem;
    address usdd;
    uint8 decimalsGem;
    uint8 decimalsUsdd;
    uint256 gemReserve;
    uint256 usddReserve;
    string gemName;
    string usddName;
    string gemSymbol;
    string usddSymbol;
    uint256 to18ConversionFactor;
    uint256 tin;
    uint256 tout;
    bool buyGemHalted;
    bool sellGemHalted;
}

struct FluidDexLiteDexIdExistResponse {
    bool isExist;
}

struct FluidDexLiteResponse{
    uint blockNumber;
    uint8 decimals0;
    uint8 decimals1;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint256 fee;
    uint256 revenueCut;
    bytes32 salt;
    uint256 token0ImaginaryReserves;
    uint256 token1ImaginaryReserves;
    uint256 token0AdjustedSupply;
    uint256 token1AdjustedSupply;
    uint256 centerPrice;
    string dexId;
}

struct MooniswapResponse {
    uint blockNumber;
    address token0;
    address token1;
    uint256 reserve0;
    uint256 reserve1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint8 decimals0;
    uint8 decimals1;
    uint256 fee;
    uint256 decayPeriod;
    uint216 vbAddition0Balance;
    uint40  vbAddition0Time;
    uint216 vbRemoval0Balance;
    uint40  vbRemoval0Time;
    uint216 vbAddition1Balance;
    uint40  vbAddition1Time;
    uint216 vbRemoval1Balance;
    uint40  vbRemoval1Time;
    uint256 slippageFee;
    uint104 virtualFeeOldResult;
    uint104 virtualFeeResult;
    uint48 virtualFeeTime;
    uint104 virtualSlippageFeeOldResult;
    uint104 virtualSlippageFeeResult;
    uint48 virtualSlippageFeeTime;
    uint104 virtualDecayPeriodOldResult;
    uint104 virtualDecayPeriodResult;
    uint48 virtualDecayPeriodTime;
}

// WooFi V2 Reader response types
struct WooFiV2TokenInfoResponse {
    uint256 reserve;
    uint16 feeRate;
    uint256 maxGamma;
    uint256 maxNotionalSwap;
    uint256 capBal;
}

struct WooFiV2StateResponse {
    uint256 price;
    uint256 spread;
    uint256 coeff;
    bool woFeasible;
}

struct WooFiV2DecimalInfoResponse {
    uint64 priceDec;
    uint64 quoteDec;
    uint64 baseDec;
}

struct WooFiV2BaseTokenDataResponse {
    WooFiV2StateResponse state;
    WooFiV2TokenInfoResponse tokenInfo;
    WooFiV2DecimalInfoResponse decimalInfo;
    string name;
    string symbol;
}

struct WooFiV2PoolsResponse {
    uint256 blockNumber;
    address poolAddress;
    address quoteToken;
    address wooracle;
    WooFiV2TokenInfoResponse quoteTokenInfo;
    string quoteTokenName;
    string quoteTokenSymbol;
    address[] baseTokenAddresses;
    WooFiV2BaseTokenDataResponse[] baseTokensData;
}

struct WooFiV2PoolsResponseV2 {
    uint256 blockNumber;
    address poolAddress;
    address quoteToken;
    address wooracle;
    WooFiV2TokenInfoResponse quoteTokenInfo;
    string quoteTokenName;
    string quoteTokenSymbol;
    address[] baseTokenAddresses;
    WooFiV2BaseTokenDataResponse[] baseTokensData;
    bool paused;
}


struct TesseraVOrderInfo {
    uint160 amount;
    uint64 priceMultiplierPpm;
    bool isDead;
}

struct TesseraVCurveState {
    TesseraVOrderInfo[20] baseToQuoteOrders;
    TesseraVOrderInfo[20] quoteToBaseOrders;
}

struct TesseraVStalenessWidenParams {
    uint64 cutoffBlock;
    uint64 widenStartBlock;
    uint32 widenCoeff;
    uint32 maxWidenPpm;
}

struct TesseraVPoolStateInfo {
    uint256 cumulativeQuoteAmountInSinceUpd;
    uint256 cumulativeBaseAmountInSinceUpd;
    uint32 widenPpm;
    uint32 cumulativeSwapAmountMultiplierPpm;
    uint8 fastPullLevel;
    bool reduceAccumulator;
    TesseraVStalenessWidenParams stalenessWidenParams;
    uint32 widenlistedPpm;
    bool enforceWhitelist;
    uint32 prioFeeThrehsoldFactor;
    uint32 widenPrioFeePpm;
    TesseraVCurveState curveState;
}

struct TesseraVSeqBlockPrice {
    uint8 seqNo;
    uint40 blockNumber;
    uint128 price;
}

struct TesseraVPoolInfo {
    // token info
    address token0;  // base token
    address token1;  // quote token
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint8 decimals0;
    uint8 decimals1;
    uint256 reserve0;
    uint256 reserve1;
    TesseraVPoolStateInfo poolState;
    TesseraVSeqBlockPrice seqBlockPrice;
}

struct TesseraVAllPoolsInfoResponse {
    uint256 blockNumber;
    uint256 globalPrioFeeThreshold;
    uint256 fastPullBlock;
    address[] pools;
    TesseraVPoolInfo[] poolInfos;
}

struct TesseraVPoolInfoV2 {
    address token0;
    address token1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint8 decimals0;
    uint8 decimals1;
    uint256 reserve0;
    uint256 reserve1;
    TesseraVPoolStateInfo poolState;
    TesseraVSeqBlockPrice seqBlockPrice;
    uint256 vaultBalance0;
    uint256 vaultBalance1;
}

struct TesseraVAllPoolsInfoResponseV2 {
    uint256 blockNumber;
    uint256 globalPrioFeeThreshold;
    uint256 fastPullBlock;
    address[] pools;
    TesseraVPoolInfoV2[] poolInfos;
}

struct DopplerHookInitializerResponse {
    address dopplerHook;
    uint256 isDopplerHookEnabled;
    uint24 hookCustomFee;
}

struct ZoraV4HookResponse {
    address coin;
    uint256 creationTimestamp;
    bool isDeploying;
    bool hasCreationInfo;
    uint8 coinType;
}

struct AlphixHookResponse {
    bool paused;
    int24 tickLower;
    int24 tickUpper;
    address yieldSource0;
    address yieldSource1;
    uint256 amount0InYieldSource;
    uint256 amount1InYieldSource;
    uint256 maxDeposit0;
    uint256 maxDeposit1;
    uint256 maxWithdraw0;
    uint256 maxWithdraw1;
}

struct ExtraPollingERC4626Response {
    uint256 blockNumber;
    uint256[] rates;
}

struct ExtraPollingRateProviderResponse {
    uint256 blockNumber;
    uint256[] rates;
}

struct ExtraPollingChainlinkResponse {
    uint256 blockNumber;
    int256[] answers;
    uint8[]  decimals;
}

/// @notice 单个探针档位：(amountIn, amountOut)
struct ElfomoFiCumulativeLevel {
    uint256 amountIn;
    uint256 amountOut;
}

/// @notice ElfomoFi 交易对 orderbook 数据 + token 元数据
struct ElfomoFiResponse {
    uint256 blockNumber;
    // token 信息
    address base;
    address quote;
    uint8   baseDecimals;
    uint8   quoteDecimals;
    string  baseName;
    string  quoteName;
    string  baseSymbol;
    string  quoteSymbol;
    // 探针数据
    ElfomoFiCumulativeLevel[] askCumulativeLevels; // quote→base（用户买 base）
    ElfomoFiCumulativeLevel[] bidCumulativeLevels; // base→quote（用户卖 base）
    // Vault 余额
    uint256 balanceBase;
    uint256 balanceQuote;
}

struct OracleQuery {
    address oracle;
    bytes4  selector;
}

struct ExtraPollingOracleResponse {
    uint256 blockNumber;
    uint256[] rates;
    bool[] successes;
}

struct RebasingQuery {
    address token;
    address pool;
}

struct ExtraPollingRebasingResponse {
    uint256 blockNumber;
    uint256[] balances;
    bool[] successes;
}

struct AaveV3ReserveResponse {
    address underlying;
    address aToken;
    uint8 decimals;
    string name;
    string symbol;
    uint256 availableLiquidity;
    uint256 supplyCap;
    uint256 totalAToken;
    uint256 totalVariableDebt;
    uint128 liquidityIndex;
    uint256 ltv;
    uint256 liquidationThreshold;
    bool isActive;
    bool isFrozen;
    bool isPaused;
    bool borrowingEnabled;
    uint8 aTokenDecimals;
    string aTokenName;
    string aTokenSymbol;
}

struct AaveV3PoolsResponse {
    uint256 blockNumber;
    address pool;
    address oracle;
    address dataProvider;
    AaveV3ReserveResponse reserve;
}

struct PancakeInfStableResponse {
    uint256 blockNumber;
    uint256 nCoins;
    address[] tokens;
    uint256[] decimals;
    uint256[] balances;
    string[] names;
    string[] symbols;
    uint256 fee;
    uint256 offpegFeeMultiplier;
    uint256[] storedRates;
    uint256 futureATime;
    uint256 futureA;
    uint256 initialATime;
    uint256 initialA;
}

struct PancakeInfStableIsPoolResponse {
    bool isPool;
    uint256 nCoins;
}

struct BalancerV3Erc4626BalanceItem {
    address wrappedToken;           // ERC4626 地址
    bool    isBufferInitialized;    // vault 是否已初始化该 token 的 buffer
    uint256 maxDeposit;             // IERC4626(wrappedToken).maxDeposit(vault)
    uint256 maxRedeem;              // IERC4626(wrappedToken).maxRedeem(vault)
    uint256 bufferUnderlyingRaw;    // vault buffer 中 underlying 数量
    uint256 bufferWrappedRaw;       // vault buffer 中 wrapped 数量
}

struct BalancerV3Erc4626BalanceResponse {
    uint256 blockNumber;
    address vault;
    bool    areBuffersPaused;       // vault 级别，任一为 true 则整批拒绝
    BalancerV3Erc4626BalanceItem[] items;
}

struct FluidExtraLiquidityResponse {
    uint256 tokenData;
    uint256 blockNumber;
}
struct Erc4626Response {
    uint256 blockNumber;
    address vault;
    uint8   vaultDecimals;
    string  vaultName;
    string  vaultSymbol;
    address underlying;
    uint8   underlyingDecimals;
    string  underlyingName;
    string  underlyingSymbol;
    uint256 totalSupply;
    uint256 totalAssets;
    uint256 maxDeposit;
    uint256 maxMint;
}

struct ExtraPollingLlamaOracleResponse {
    uint256 blockNumber;
    uint256[] oraclePrices;
    uint256[] basePrices;
    bool[] successes;
}

// ─── Curve LlamaLend (LLAMMA AMM) ─────────────────────────────────────────────

struct CurveLlamaLendBand {
    int256  n;
    uint256 x;
    uint256 y;
}

struct CurveLlamaLendResponse {
    uint256 blockNumber;
    address token0;
    address token1;
    uint8   decimals0;
    uint8   decimals1;
    string  name0;
    string  name1;
    string  symbol0;
    string  symbol1;
    int256  minBand;
    int256  maxBand;
    int256  activeBand;
    uint256 A;
    uint256 fee1e18;
    uint256 basePrice;
    uint256 oraclePrice;
    uint256 pOracleUpActive;
    CurveLlamaLendBand[] bands;
}

struct OriginArmResponse {
    uint256 blockNumber;
    address token0;
    address token1;
    uint8   decimals0;
    uint8   decimals1;
    string  name0;
    string  name1;
    string  symbol0;
    string  symbol1;
    uint256 traderate0;       // token0→token1 price, 1e36 scale
    uint256 traderate1;       // token1→token0 price, 1e36 scale
    uint256 crossPrice;       // buy/sell price boundary, 1e36 scale
    uint256 reserve0;         // available token0 liquidity (after deducting withdrawal queue)
    uint256 reserve1;         // available token1 liquidity
    address liquidityAsset;   // LP deposit/withdraw asset (WETH etc.)
    address baseAsset;        // asset purchased & redeemed (stETH etc.)
    uint256 withdrawsQueued;  // total queued liquidity withdrawals
    uint256 withdrawsClaimed; // total claimed liquidity withdrawals
}

struct LiquidCoreResponse {
    uint256 blockNumber;
    address token0;
    address token1;
    uint8 decimals0;
    uint8 decimals1;
    string name0;
    string name1;
    string symbol0;
    string symbol1;
    uint256 reserve0;
    uint256 reserve1;
    // token0→token1 方向：tick[i]=token0 amountIn，tickEstimatePrice[i]=token1 amountOut
    // 等差 tickSize 份，范围 [1×10^dec0, maxInputAmount0]
    uint256[] tick;
    uint256[] tickEstimatePrice;
    // token1→token0 方向：tick1[i]=token1 amountIn，tick1EstimatePrice[i]=token0 amountOut
    // 等差 tickSize 份，范围 [1×10^dec1, maxInputAmount1]
    uint256[] tick1;
    uint256[] tick1EstimatePrice;
}

struct ObricV2Response {
    uint256 blockNumber;
    address token0;
    address token1;
    uint8   decimals0;
    string  name0;
    string  symbol0;
    uint8   decimals1;
    string  name1;
    string  symbol1;
    uint256 reserveX;
    uint256 reserveY;
    uint256 currentXK;
    uint256 preK;
    uint256 multYBase;
    uint64  feeMillionth;
    uint256 priceMaxAge;
    uint256 priceUpdateTime;
    bool    isLocked;
}


// ===== FILE: src/types/request.sol =====
pragma solidity ^0.8.13;

// !!! 注意：已经上生产的枚举，不要随意修改顺序
enum Dex {
    UNISWAP_V2,
    UNISWAP_V3,
    PANCAKESWAP_V3,
    PANCAKESWAP_V2,
    DODO_V2,
    UNISWAP_V4,
    PANCAKE_STABLE,
    PANCAKE_INF_CL,
    PANCAKE_INF_BIN,
    WOMBAT,
    ALGEBRA_V1,
    THENA_INTEGRAL,
    FLUID_DEX,
    SLIPSTREAM,
    AERODROME_VOLATILE,
    AERODROME_STABLE,
    CURVE_STABLE_NG,
    CURVE_STABLE_V1,
    QUICKSWAP_V4,
    EKUBO,
    CAMELOT_V2,
    ALGEBRA_V1_DIR_FEE,
    CAMELOT_V4,
    CURVE_CRYPTO,
    CURVE_CRYPTO_NG,
    BALANCER_WEIGHTED_V2,
    BALANCER_STABLE_V2,
    SQUAD_SWAP_V2,
    SQUAD_SWAP_V3,
    LIDO,
    LFJ_V2_LEGACY,
    LFJ_V2,
    RAMSES_V3,
    FLUID_DEX_LITE,
    EKUBO_V3,
    SHIBA_SWAP_V1,
    W_DEX,
    CYPHER_V4,
    INTEGRAL,
    HYDREX_V4,
    SOLIDLY_V3,
    SUPERNOVA_CL,
    TREBLE_SWAP,
    TREBLE_SWAP_V2
}

struct UniswapV2LikeRequest {
    address pool;
    Dex dex;
}

struct UniswapV3LikeRequest {
    address pool;
    Dex dex;
    int16 bitmapIndexLimit;
}

struct UniswapV3LikeTicksRequest {
    address pool;
    Dex dex;
    int16 bitmapOffsetStart;
    int16 bitmapOffsetEnd;
}

struct DoDoV2Request {
    address pool;
    Dex dex;
    address router;
}

struct PancakeStableRequest {
    address pool;
    Dex dex;
}

struct UniswapV4PoolKey {
    address currency0;
    address currency1;
    uint24 fee;
    int24 tickSpacing;
    address hooks;
}

struct UniswapV4LikeV3Request {
    address poolManager;
    bytes32 poolId;
    Dex dex;
    int16 bitmapIndexLimit;
    UniswapV4PoolKey poolKey;
}

struct UniswapV4LikeRequest {
    address poolManager;
    address positionManager;
    bytes32 poolId;
    Dex dex;
    int16 bitmapIndexLimit;
}

struct UniswapV4LikeTicksRequest {
    address poolManager;
    address positionManager;
    bytes32 poolId;
    Dex dex;
    int16 bitmapOffsetStart;
    int16 bitmapOffsetEnd;
}

struct PancakeInfBinRequest {
    address poolManager;
    bytes32 poolId;
    Dex dex;
    uint24 binRangeLimit;
}

struct PancakeInfBinBinsRequest {
    address poolManager;
    bytes32 poolId;
    Dex dex;
    int24 binOffsetStart;
    int24 binOffsetEnd;
}

struct WombatRequest {
    address pool;
    Dex dex;
}

struct AlgebraV1LikeRequest {
    address pool;
    Dex dex;
    int16 bitmapIndexLimit;
    uint32 timepointsSampleInterval;
}

struct AlgebraV1LikeRequestV2 {
    address pool;
    Dex dex;
    int16 bitmapIndexLimit;
    address quoter;
}

struct AlgebraV1LikeTicksRequest {
    address pool;
    Dex dex;
    int16 bitmapOffsetStart;
    int16 bitmapOffsetEnd;
}

struct AlgebraIntegralLikeRequest {
    address pool;
    Dex dex;
    uint24 tickCountLimit;
    uint32 timepointsSampleInterval;
}

struct SupernovaPoolInfoRequest {
    address pool;
    uint24 tickCountLimit;
    uint32 timepointsSampleInterval;
}

struct WDexRequest {
    address pool;
    uint24 tickCountLimit;
    uint32 timepointsSampleInterval;
}

struct FluidDexRequest {
    address pool;
    Dex dex;
    address dexReservesResolver;
}

enum ClankerHookType {
    STATIC_FEE_V2,
    DYNAMIC_FEE_V2,
    FEY_STATIC_FEE_V2,
    STATIC_FEE,
    DYNAMIC_FEE,
    LIQUID_STATIC_FEE_V2,
    LIQUID_DYNAMIC_FEE_V2
}

struct ClankerHookV2RequestV2 {
    bytes32 poolId;
    address hookAddress;
    ClankerHookType hookType;
    UniswapV4PoolKey poolKey;
}

struct SlipstreamRequest {
    address pool;
    Dex dex;
    int16 bitmapIndexLimit;
    uint32 observationSize;
    uint32 observationSecondsAgo;
}


struct CurveFactoryRequest {
    address metaRegistry;
    address pool;
    Dex dex;
}

struct BalancerV2WeightedRequest {
    address pool;
    address vault;
    Dex dex;
}

struct BalancerV1PoolRequest {
    address pool;
}

struct BalancerV2StableRequest {
    address pool;
    address vault;
    Dex dex;
}

struct BalancerV2PoolTypeRequest {
    address vault;
    address pool;
}

struct BalancerV2EclpRequest {
    address pool;
    address vault;
}


struct BalancerV3BasePoolRequest {
    address vault;
    address pool;
}

struct LidoRequest {
    address pool;
    Dex dex;
}

struct EkuboPoolKey {
    address token0;
    address token1;
    address extension;
    uint64 fee;
    uint32 tickSpacing;
}

struct EkuboPoolInfoRequest {
    address poolManager;
    bytes32 poolId;
    Dex dex;
    EkuboPoolKey poolKey;
    int16 bitmapIndexLimit;
    uint256 skipAhead;
}

/// @notice Request structure for Ekubo V3 pool information
/// @dev V3 uses PoolConfig (bytes32) instead of separate extension/fee/tickSpacing fields
struct EkuboPoolInfoRequestV3 {
    address core;
    bytes32 poolId;
    Dex dex;
    EkuboV3PoolKey poolKey;
    int16 bitmapIndexLimit;
    uint256 skipAhead;
}

/// @notice Pool key structure for Ekubo V3
/// @dev V3 uses a packed PoolConfig (bytes32) instead of separate fields
struct EkuboV3PoolKey {
    /// @notice Address of token0 (must be < token1)
    address token0;
    /// @notice Address of token1 (must be > token0)
    address token1;
    /// @notice Packed pool configuration (bytes32)
    /// @dev Contains: extension (20B) + fee (8B) + type config (4B)
    ///      Type config bit 31: 1 = Concentrated, 0 = Stableswap
    ///      For Concentrated: bits 30-0 = tick spacing
    ///      For Stableswap: bits 30-24 = amplification, bits 23-0 = center tick
    bytes32 config;
}

struct UniswapV4HookRequest {
    bytes32 poolId;
    address positionManager;
    address hookAddress;
    UniswapV4PoolKey poolKey;
}

struct BancorV2Request {
    address poolAddress; // Converter 地址（从事件 log.Address 获取）
    address converterRegistry; // ConverterRegistry 地址，用于验证 converter 是否有效
}

struct BancorV3Request {
    address poolCollection; // PoolCollection 合约地址
    address pool; // 池子地址（代币地址，TKN）
    address bancorNetworkInfo; // BancorNetworkInfo 合约地址
}

struct LfjV2Request {
    address pool;
    Dex dex;
    uint24 binCountLimit;
    address hooksManager;
}

struct NomiswapStableRequest {
    address pool;
}

struct TesseraVPoolInfoRequest {
    address pool;
    address manager;
}

struct FluidDexLiteDexIdRequest {
    bytes8 dexId;
    address fluidDexLiteResolver;
}

struct FluidDexLitePoolRequest {
    bytes8 dexId;
    address fluidDexLiteResolver;
    address fluidDexLite;
}


struct MaverickV2Request {
    address pool;
    address lens;
    int32 tickRadius;
}

struct MaverickV2ExtraTicksRequest {
    address pool;
    address lens;
    int32 tickRadiusStart;
    int32 tickRadiusEnd;
}

struct SolidlyV3PoolInfoRequest {
    address pool;
    int16 bitmapIndexLimit;
}

struct SolidlyV3TicksRequest {
    address pool;
    int16 bitmapOffsetStart;
    int16 bitmapOffsetEnd;
}

/// @notice Request for ElfomoFi pool orderbook data
struct ElfomoFiRequest {
    address helper;  // ElfomoFiHelper 合约地址（每链独立部署）
    address base;    // base token 地址
    address quote;   // quote token 地址
}

struct PancakeInfStableRequest {
    address pool;
    address factory;
}

struct FluidExtraLiquidityRequest {
    address liquidity;
    bytes32 slot;
}

struct OriginArmRequest {
    address pool;
}


// ===== FILE: src/common/IBalancerV3Reader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {BalancerV3BasePoolRequest} from "../types/request.sol";
import {
    BalancerV3BasePoolResponse,
    BalancerV3BasePoolResponseV2,
    BalancerV3PoolType,
    BalancerV3PoolFeatures,
    BalancerV3Erc4626BalanceResponse
} from "../types/response.sol";

interface IBalancerV3Reader {
    function balancerV3PoolType(BalancerV3BasePoolRequest calldata request) external view returns (BalancerV3PoolType);
    function balancerV3PoolFeatures(BalancerV3BasePoolRequest calldata request) external view returns (BalancerV3PoolFeatures memory features);
    function balancerV3BasePoolInfoV2(BalancerV3BasePoolRequest calldata request) external view returns (BalancerV3BasePoolResponseV2 memory response);
    function batchBalancerV3Erc4626Balance(address vault, address[] calldata rateProviders)
        external
        view
        returns (BalancerV3Erc4626BalanceResponse memory response);
}

// ===== FILE: src/common/IBalancerV1Reader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {BalancerV1PoolRequest} from "../types/request.sol";
import {BalancerV1PoolResponse} from "../types/response.sol";

interface IBalancerV1Reader {
    function balancerV1PoolInfo(BalancerV1PoolRequest calldata params)
        external
        view
        returns (BalancerV1PoolResponse memory response);

    /// @return true if pool was created by the given BFactory
    function isBalancerV1Pool(address pool, address factory) external view returns (bool);
}


// ===== FILE: src/common/IExtraPollingReader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import "../types/response.sol";
import {ExtraPollingERC4626Response, ExtraPollingRateProviderResponse, ExtraPollingChainlinkResponse, OracleQuery, ExtraPollingOracleResponse, RebasingQuery, ExtraPollingRebasingResponse, ExtraPollingLlamaOracleResponse} from "../types/response.sol";

interface IExtraPollingReader {
    function batchERC4626ConvertToAssets(address[] calldata tokens) external view returns (ExtraPollingERC4626Response memory result);
    function batchRateProviderGetRate(address[] calldata rateProviders) external view returns (ExtraPollingRateProviderResponse memory result);
    function batchChainlinkOracleData(address[] calldata oracles) external view returns (ExtraPollingChainlinkResponse memory result);
    function batchOracleRate(OracleQuery[] calldata queries) external view returns (ExtraPollingOracleResponse memory result);
    function batchRebasingBalances(RebasingQuery[] calldata queries) external view returns (ExtraPollingRebasingResponse memory result);
    function batchLlamaOracleRate(address[] calldata tokens) external view returns (ExtraPollingLlamaOracleResponse memory result);
}

// ===== FILE: src/common/Erc4626Reader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {IErc4626Reader} from "./IErc4626Reader.sol";
import {IERC4626Minimal} from "../abis/extra_polling.sol";
import {Library} from "./Library.sol";
import {Erc4626Response} from "../types/response.sol";

interface IERC20TotalSupply {
    function totalSupply() external view returns (uint256);
}

contract Erc4626Reader is IErc4626Reader {
    function erc4626PoolInfo(address vault, address handler)
        external
        view
        returns (Erc4626Response memory result)
    {
        result.blockNumber = block.number;
        result.vault = vault;

        // vault token info
        (result.vaultDecimals, result.vaultName, result.vaultSymbol) = Library.getTokenInfo(vault);

        // underlying asset address
        try IERC4626Minimal(vault).asset() returns (address underlying) {
            result.underlying = underlying;
            (result.underlyingDecimals, result.underlyingName, result.underlyingSymbol) = Library.getTokenInfo(underlying);
        } catch {
            return result;
        }

        try IERC20TotalSupply(vault).totalSupply() returns (uint256 ts) {
            result.totalSupply = ts;
        } catch {}

        try IERC4626Minimal(vault).totalAssets() returns (uint256 ta) {
            result.totalAssets = ta;
        } catch {}

        try IERC4626Minimal(vault).maxDeposit(handler) returns (uint256 md) {
            result.maxDeposit = md;
        } catch {}

        try IERC4626Minimal(vault).maxMint(handler) returns (uint256 mm) {
            result.maxMint = mm;
        } catch {}
    }
}


// ===== FILE: src/common/ICurveLlamaLendReader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {CurveLlamaLendResponse} from "../types/response.sol";

interface ICurveLlamaLendReader {
    function curveLlamaLendPoolInfo(address poolAddr, int256 maxBandWindow)
        external
        view
        returns (CurveLlamaLendResponse memory);

    // 批量读取所有 factory 的 AMM 地址
    function curveLlamaLendFactoryAmms(address[] calldata factories)
        external
        view
        returns (address[] memory amms);
}


// ===== FILE: src/common/IOriginArmReader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {OriginArmRequest} from "../types/request.sol";
import {OriginArmResponse} from "../types/response.sol";

interface IOriginArmReader {
    function originArmPoolInfo(OriginArmRequest calldata params) external view returns (OriginArmResponse memory response);
}


// ===== FILE: src/common/IObricV2Reader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {ObricV2Response} from "../types/response.sol";

interface IObricV2Reader {
    function obricV2PoolInfo(address poolAddr) external view returns (ObricV2Response memory response);
}


// ===== FILE: src/common/IDodoV1Reader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {DoDoV1Response} from "../types/response.sol";

interface IDodoV1Reader {
    function dodoV1PoolInfo(address pool) external view returns (DoDoV1Response memory response);
}


// ===== FILE: src/common/IDodoGspReader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {DoDoGspResponse} from "../types/response.sol";

interface IDodoGspReader {
    function dodoGspPoolInfo(address pool) external view returns (DoDoGspResponse memory response);
}


// ===== FILE: src/common/IEkuboReader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {EkuboPoolInfoRequest,EkuboPoolInfoRequestV3} from "../types/request.sol";
import {EkuboPoolInfoResponse} from "../types/response.sol";

interface IEkuboReader {
    /// @notice Read Ekubo V2 pool information including state, tokens, and ticks
    /// @param params Request parameters containing pool manager, pool ID, pool key, and bitmap limits
    /// @return response Full pool information
    function ekuboPoolInfo(EkuboPoolInfoRequest calldata params) external view returns (EkuboPoolInfoResponse memory response);
    
    /// @notice Read Ekubo V3 pool information including state, tokens, ticks, and pool type
    /// @param params Request parameters containing Core contract address, pool ID, pool key (with PoolConfig), and bitmap limits
    /// @return response Full pool information with V3-specific fields (poolType, amplification, centerTick)
    /// @dev V3 uses Singleton architecture with Core contract and packed PoolConfig (bytes32)
    /// @dev Supports three pool types: Concentrated, Stableswap, and Full Range
    function ekuboPoolInfoV3(EkuboPoolInfoRequestV3 calldata params) external view returns (EkuboPoolInfoResponse memory response);
}



// ===== FILE: src/abis/balancer_v3_vault.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

struct HooksConfig {
    bool enableHookAdjustedAmounts;
    bool shouldCallBeforeInitialize;
    bool shouldCallAfterInitialize;
    bool shouldCallComputeDynamicSwapFee;
    bool shouldCallBeforeSwap;
    bool shouldCallAfterSwap;
    bool shouldCallBeforeAddLiquidity;
    bool shouldCallAfterAddLiquidity;
    bool shouldCallBeforeRemoveLiquidity;
    bool shouldCallAfterRemoveLiquidity;
    address hooksContract;
}

enum TokenType {
    STANDARD,
    WITH_RATE
}

struct TokenInfo {
    TokenType tokenType;
    address rateProvider;
    bool paysYieldFees;
}

struct PoolData {
    bytes32 poolConfigBits;
    address[] tokens;
    TokenInfo[] tokenInfo;
    uint256[] balancesRaw;
    uint256[] balancesLiveScaled18;
    uint256[] tokenRates;
    uint256[] decimalScalingFactors;
}

enum AddLiquidityKind {
    PROPORTIONAL,
    UNBALANCED,
    SINGLE_TOKEN_EXACT_OUT,
    DONATION,
    CUSTOM
}

enum RemoveLiquidityKind {
    PROPORTIONAL,
    SINGLE_TOKEN_EXACT_IN,
    SINGLE_TOKEN_EXACT_OUT,
    CUSTOM
}

interface Vault {
    event Swap(
        address indexed pool,
        address indexed tokenIn,
        address indexed tokenOut,
        uint256 amountIn,
        uint256 amountOut,
        uint256 swapFeePercentage,
        uint256 swapFeeAmount
    );

    event LiquidityAdded(
        address indexed pool,
        address indexed liquidityProvider,
        AddLiquidityKind indexed kind,
        uint256 totalSupply,
        uint256[] amountsAddedRaw,
        uint256[] swapFeeAmountsRaw
    );

    event LiquidityRemoved(
        address indexed pool,
        address indexed liquidityProvider,
        RemoveLiquidityKind indexed kind,
        uint256 totalSupply,
        uint256[] amountsRemovedRaw,
        uint256[] swapFeeAmountsRaw
    );

    event PoolPausedStateChanged(address indexed pool, bool paused);
    event SwapFeePercentageChanged(address indexed pool, uint256 swapFeePercentage);
    event VaultAuxiliary(address indexed pool, bytes32 indexed eventKey, bytes eventData);

    /// @notice The user tried to swap zero tokens.
    error AmountGivenZero();

    /// @notice The user attempted to swap a token for itself.
    error CannotSwapSameToken();

    /**
     * @notice The user attempted to operate with a token that is not in the pool.
     * @param token The unregistered token
     */
    error TokenNotRegistered(address token);

    /**
     * @notice An amount in or out has exceeded the limit specified in the swap request.
     * @param amount The total amount in or out
     * @param limit The amount of the limit that has been exceeded
     */
    error SwapLimit(uint256 amount, uint256 limit);

    /**
     * @notice A hook adjusted amount in or out has exceeded the limit specified in the swap request.
     * @param amount The total amount in or out
     * @param limit The amount of the limit that has been exceeded
     */
    error HookAdjustedSwapLimit(uint256 amount, uint256 limit);

    /// @notice The amount given or calculated for an operation is below the minimum limit.
    error TradeAmountTooSmall();

    function isVaultPaused() external view returns (bool vaultPaused);
    function isPoolInitialized(address pool) external view returns (bool initialized);
    function isPoolPaused(address pool) external view returns (bool poolPaused);

    function getPoolTokens(address pool) external view returns (address[] memory tokens);
    function getPoolData(address pool) external view returns (PoolData memory poolData);
    function getPoolTokenInfo(address pool)
        external
        view
        returns (
            address[] memory tokens,
            TokenInfo[] memory tokenInfo,
            uint256[] memory balancesRaw,
            uint256[] memory lastBalancesLiveScaled18
        );

    function getStaticSwapFeePercentage(address pool) external view returns (uint256 swapFeePercentage);
    function getHooksConfig(address pool) external view returns (HooksConfig memory hooksConfig);

    function areBuffersPaused() external view returns (bool);
    function getBufferBalance(address wrappedToken)
        external
        view
        returns (uint256 underlyingBalanceRaw, uint256 wrappedBalanceRaw);
    function isERC4626BufferInitialized(address wrappedToken) external view returns (bool);
}

// ===== FILE: src/common/IErc4626Reader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {Erc4626Response} from "../types/response.sol";

interface IErc4626Reader {
    function erc4626PoolInfo(address vault, address handler) external view returns (Erc4626Response memory result);
}


// ===== FILE: src/abis/extra_polling.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

interface IERC4626Minimal {
    event Deposit(address indexed sender, address indexed owner, uint256 assets, uint256 shares);

    event Withdraw(
        address indexed sender,
        address indexed receiver,
        address indexed owner,
        uint256 assets,
        uint256 shares
    );
    function asset() external view returns (address);
    function decimals() external view returns (uint8);
    function convertToAssets(uint256 shares) external view returns (uint256);
    function totalAssets() external view returns (uint256);
    function maxDeposit(address receiver) external view returns (uint256);
    function maxMint(address receiver) external view returns (uint256);
    function maxRedeem(address owner) external view returns (uint256);
    function maxWithdraw(address owner) external view returns (uint256);
}

interface IRateProvider {
    function getRate() external view returns (uint256);
}

interface IERC4626RateProvider is IRateProvider {
    function erc4626() external view returns (address);
}

interface IChainlinkRateProvider is IRateProvider {
    function pricefeed() external view returns (address);
}

interface IAaveMarketRateTransformer is IERC4626RateProvider {
    function vaultAssetFeed() external view returns (address);
}

interface IERC20Decimals {
    function decimals() external view returns (uint8);
}

interface IAggregatorV3 {
    function decimals() external view returns (uint8);
    function latestRoundData()
        external
        view
        returns (
            uint80 roundId,
            int256 answer,
            uint256 startedAt,
            uint256 updatedAt,
            uint80 answeredInRound
        );
}

// ===== FILE: src/common/Library.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {ERC20} from "openzeppelin-contracts/contracts/token/ERC20/ERC20.sol";

library Library {
    function bytesToString(bytes memory data) internal pure returns (string memory) {
        if (data.length != 32 || data[0] == 0) {
            return abi.decode(data, (string));
        }

        uint256 len = 0;

        // 计算实际字符串长度（遇到 0x00 停止）
        while (len < 32 && data[len] != 0) {
            len++;
        }

        bytes memory bytesArray = new bytes(len);

        for (uint256 i = 0; i < len; i++) {
            bytesArray[i] = data[i];
        }

        return string(bytesArray);
    }

    function getTokenInfo(address token) internal view returns (uint8 decimals, string memory name, string memory symbol) {
        if (token != address(0)) {
            ERC20 tokenContract = ERC20(token);
            decimals = tokenContract.decimals();

            bool success;
            bytes memory data;
            (success, data) = token.staticcall(abi.encodeWithSignature("name()"));
            if (success) {
                name = bytesToString(data);
            } else {
                name = "";
            }
            (success, data) = token.staticcall(abi.encodeWithSignature("symbol()"));
            if (success) {
                symbol = bytesToString(data);
            } else {
                symbol = "";
            }
        } else {
            decimals = 18;
            name = "ETH";
            symbol = "ETH";
        }
    }

    function isAToken(address token) internal view returns (bool) {
        (bool success, bytes memory data) = token.staticcall(abi.encodeWithSignature("UNDERLYING_ASSET_ADDRESS()"));
        if (success && data.length == 32) {
            address underlying = abi.decode(data, (address));
            return underlying != address(0);
        }
        return false;
    }
}

// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/token/ERC20/ERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (token/ERC20/ERC20.sol)

pragma solidity ^0.8.20;

import {IERC20} from "./IERC20.sol";
import {IERC20Metadata} from "./extensions/IERC20Metadata.sol";
import {Context} from "../../utils/Context.sol";
import {IERC20Errors} from "../../interfaces/draft-IERC6093.sol";

/**
 * @dev Implementation of the {IERC20} interface.
 *
 * This implementation is agnostic to the way tokens are created. This means
 * that a supply mechanism has to be added in a derived contract using {_mint}.
 *
 * TIP: For a detailed writeup see our guide
 * https://forum.openzeppelin.com/t/how-to-implement-erc20-supply-mechanisms/226[How
 * to implement supply mechanisms].
 *
 * The default value of {decimals} is 18. To change this, you should override
 * this function so it returns a different value.
 *
 * We have followed general OpenZeppelin Contracts guidelines: functions revert
 * instead returning `false` on failure. This behavior is nonetheless
 * conventional and does not conflict with the expectations of ERC-20
 * applications.
 */
abstract contract ERC20 is Context, IERC20, IERC20Metadata, IERC20Errors {
    mapping(address account => uint256) private _balances;

    mapping(address account => mapping(address spender => uint256)) private _allowances;

    uint256 private _totalSupply;

    string private _name;
    string private _symbol;

    /**
     * @dev Sets the values for {name} and {symbol}.
     *
     * Both values are immutable: they can only be set once during construction.
     */
    constructor(string memory name_, string memory symbol_) {
        _name = name_;
        _symbol = symbol_;
    }

    /**
     * @dev Returns the name of the token.
     */
    function name() public view virtual returns (string memory) {
        return _name;
    }

    /**
     * @dev Returns the symbol of the token, usually a shorter version of the
     * name.
     */
    function symbol() public view virtual returns (string memory) {
        return _symbol;
    }

    /**
     * @dev Returns the number of decimals used to get its user representation.
     * For example, if `decimals` equals `2`, a balance of `505` tokens should
     * be displayed to a user as `5.05` (`505 / 10 ** 2`).
     *
     * Tokens usually opt for a value of 18, imitating the relationship between
     * Ether and Wei. This is the default value returned by this function, unless
     * it's overridden.
     *
     * NOTE: This information is only used for _display_ purposes: it in
     * no way affects any of the arithmetic of the contract, including
     * {IERC20-balanceOf} and {IERC20-transfer}.
     */
    function decimals() public view virtual returns (uint8) {
        return 18;
    }

    /// @inheritdoc IERC20
    function totalSupply() public view virtual returns (uint256) {
        return _totalSupply;
    }

    /// @inheritdoc IERC20
    function balanceOf(address account) public view virtual returns (uint256) {
        return _balances[account];
    }

    /**
     * @dev See {IERC20-transfer}.
     *
     * Requirements:
     *
     * - `to` cannot be the zero address.
     * - the caller must have a balance of at least `value`.
     */
    function transfer(address to, uint256 value) public virtual returns (bool) {
        address owner = _msgSender();
        _transfer(owner, to, value);
        return true;
    }

    /// @inheritdoc IERC20
    function allowance(address owner, address spender) public view virtual returns (uint256) {
        return _allowances[owner][spender];
    }

    /**
     * @dev See {IERC20-approve}.
     *
     * NOTE: If `value` is the maximum `uint256`, the allowance is not updated on
     * `transferFrom`. This is semantically equivalent to an infinite approval.
     *
     * Requirements:
     *
     * - `spender` cannot be the zero address.
     */
    function approve(address spender, uint256 value) public virtual returns (bool) {
        address owner = _msgSender();
        _approve(owner, spender, value);
        return true;
    }

    /**
     * @dev See {IERC20-transferFrom}.
     *
     * Skips emitting an {Approval} event indicating an allowance update. This is not
     * required by the ERC. See {xref-ERC20-_approve-address-address-uint256-bool-}[_approve].
     *
     * NOTE: Does not update the allowance if the current allowance
     * is the maximum `uint256`.
     *
     * Requirements:
     *
     * - `from` and `to` cannot be the zero address.
     * - `from` must have a balance of at least `value`.
     * - the caller must have allowance for ``from``'s tokens of at least
     * `value`.
     */
    function transferFrom(address from, address to, uint256 value) public virtual returns (bool) {
        address spender = _msgSender();
        _spendAllowance(from, spender, value);
        _transfer(from, to, value);
        return true;
    }

    /**
     * @dev Moves a `value` amount of tokens from `from` to `to`.
     *
     * This internal function is equivalent to {transfer}, and can be used to
     * e.g. implement automatic token fees, slashing mechanisms, etc.
     *
     * Emits a {Transfer} event.
     *
     * NOTE: This function is not virtual, {_update} should be overridden instead.
     */
    function _transfer(address from, address to, uint256 value) internal {
        if (from == address(0)) {
            revert ERC20InvalidSender(address(0));
        }
        if (to == address(0)) {
            revert ERC20InvalidReceiver(address(0));
        }
        _update(from, to, value);
    }

    /**
     * @dev Transfers a `value` amount of tokens from `from` to `to`, or alternatively mints (or burns) if `from`
     * (or `to`) is the zero address. All customizations to transfers, mints, and burns should be done by overriding
     * this function.
     *
     * Emits a {Transfer} event.
     */
    function _update(address from, address to, uint256 value) internal virtual {
        if (from == address(0)) {
            // Overflow check required: The rest of the code assumes that totalSupply never overflows
            _totalSupply += value;
        } else {
            uint256 fromBalance = _balances[from];
            if (fromBalance < value) {
                revert ERC20InsufficientBalance(from, fromBalance, value);
            }
            unchecked {
                // Overflow not possible: value <= fromBalance <= totalSupply.
                _balances[from] = fromBalance - value;
            }
        }

        if (to == address(0)) {
            unchecked {
                // Overflow not possible: value <= totalSupply or value <= fromBalance <= totalSupply.
                _totalSupply -= value;
            }
        } else {
            unchecked {
                // Overflow not possible: balance + value is at most totalSupply, which we know fits into a uint256.
                _balances[to] += value;
            }
        }

        emit Transfer(from, to, value);
    }

    /**
     * @dev Creates a `value` amount of tokens and assigns them to `account`, by transferring it from address(0).
     * Relies on the `_update` mechanism
     *
     * Emits a {Transfer} event with `from` set to the zero address.
     *
     * NOTE: This function is not virtual, {_update} should be overridden instead.
     */
    function _mint(address account, uint256 value) internal {
        if (account == address(0)) {
            revert ERC20InvalidReceiver(address(0));
        }
        _update(address(0), account, value);
    }

    /**
     * @dev Destroys a `value` amount of tokens from `account`, lowering the total supply.
     * Relies on the `_update` mechanism.
     *
     * Emits a {Transfer} event with `to` set to the zero address.
     *
     * NOTE: This function is not virtual, {_update} should be overridden instead
     */
    function _burn(address account, uint256 value) internal {
        if (account == address(0)) {
            revert ERC20InvalidSender(address(0));
        }
        _update(account, address(0), value);
    }

    /**
     * @dev Sets `value` as the allowance of `spender` over the `owner`'s tokens.
     *
     * This internal function is equivalent to `approve`, and can be used to
     * e.g. set automatic allowances for certain subsystems, etc.
     *
     * Emits an {Approval} event.
     *
     * Requirements:
     *
     * - `owner` cannot be the zero address.
     * - `spender` cannot be the zero address.
     *
     * Overrides to this logic should be done to the variant with an additional `bool emitEvent` argument.
     */
    function _approve(address owner, address spender, uint256 value) internal {
        _approve(owner, spender, value, true);
    }

    /**
     * @dev Variant of {_approve} with an optional flag to enable or disable the {Approval} event.
     *
     * By default (when calling {_approve}) the flag is set to true. On the other hand, approval changes made by
     * `_spendAllowance` during the `transferFrom` operation set the flag to false. This saves gas by not emitting any
     * `Approval` event during `transferFrom` operations.
     *
     * Anyone who wishes to continue emitting `Approval` events on the`transferFrom` operation can force the flag to
     * true using the following override:
     *
     * ```solidity
     * function _approve(address owner, address spender, uint256 value, bool) internal virtual override {
     *     super._approve(owner, spender, value, true);
     * }
     * ```
     *
     * Requirements are the same as {_approve}.
     */
    function _approve(address owner, address spender, uint256 value, bool emitEvent) internal virtual {
        if (owner == address(0)) {
            revert ERC20InvalidApprover(address(0));
        }
        if (spender == address(0)) {
            revert ERC20InvalidSpender(address(0));
        }
        _allowances[owner][spender] = value;
        if (emitEvent) {
            emit Approval(owner, spender, value);
        }
    }

    /**
     * @dev Updates `owner`'s allowance for `spender` based on spent `value`.
     *
     * Does not update the allowance value in case of infinite allowance.
     * Revert if not enough allowance is available.
     *
     * Does not emit an {Approval} event.
     */
    function _spendAllowance(address owner, address spender, uint256 value) internal virtual {
        uint256 currentAllowance = allowance(owner, spender);
        if (currentAllowance < type(uint256).max) {
            if (currentAllowance < value) {
                revert ERC20InsufficientAllowance(spender, currentAllowance, value);
            }
            unchecked {
                _approve(owner, spender, currentAllowance - value, false);
            }
        }
    }
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/token/ERC20/IERC20.sol =====
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


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/token/ERC20/extensions/IERC20Metadata.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (token/ERC20/extensions/IERC20Metadata.sol)

pragma solidity >=0.6.2;

import {IERC20} from "../IERC20.sol";

/**
 * @dev Interface for the optional metadata functions from the ERC-20 standard.
 */
interface IERC20Metadata is IERC20 {
    /**
     * @dev Returns the name of the token.
     */
    function name() external view returns (string memory);

    /**
     * @dev Returns the symbol of the token.
     */
    function symbol() external view returns (string memory);

    /**
     * @dev Returns the decimals places of the token.
     */
    function decimals() external view returns (uint8);
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/utils/Context.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.1) (utils/Context.sol)

pragma solidity ^0.8.20;

/**
 * @dev Provides information about the current execution context, including the
 * sender of the transaction and its data. While these are generally available
 * via msg.sender and msg.data, they should not be accessed in such a direct
 * manner, since when dealing with meta-transactions the account sending and
 * paying for execution may not be the actual sender (as far as an application
 * is concerned).
 *
 * This contract is only required for intermediate, library-like contracts.
 */
abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }

    function _contextSuffixLength() internal view virtual returns (uint256) {
        return 0;
    }
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/interfaces/draft-IERC6093.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (interfaces/draft-IERC6093.sol)
pragma solidity >=0.8.4;

/**
 * @dev Standard ERC-20 Errors
 * Interface of the https://eips.ethereum.org/EIPS/eip-6093[ERC-6093] custom errors for ERC-20 tokens.
 */
interface IERC20Errors {
    /**
     * @dev Indicates an error related to the current `balance` of a `sender`. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     * @param balance Current balance for the interacting account.
     * @param needed Minimum amount required to perform a transfer.
     */
    error ERC20InsufficientBalance(address sender, uint256 balance, uint256 needed);

    /**
     * @dev Indicates a failure with the token `sender`. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     */
    error ERC20InvalidSender(address sender);

    /**
     * @dev Indicates a failure with the token `receiver`. Used in transfers.
     * @param receiver Address to which tokens are being transferred.
     */
    error ERC20InvalidReceiver(address receiver);

    /**
     * @dev Indicates a failure with the `spender`’s `allowance`. Used in transfers.
     * @param spender Address that may be allowed to operate on tokens without being their owner.
     * @param allowance Amount of tokens a `spender` is allowed to operate with.
     * @param needed Minimum amount required to perform a transfer.
     */
    error ERC20InsufficientAllowance(address spender, uint256 allowance, uint256 needed);

    /**
     * @dev Indicates a failure with the `approver` of a token to be approved. Used in approvals.
     * @param approver Address initiating an approval operation.
     */
    error ERC20InvalidApprover(address approver);

    /**
     * @dev Indicates a failure with the `spender` to be approved. Used in approvals.
     * @param spender Address that may be allowed to operate on tokens without being their owner.
     */
    error ERC20InvalidSpender(address spender);
}

/**
 * @dev Standard ERC-721 Errors
 * Interface of the https://eips.ethereum.org/EIPS/eip-6093[ERC-6093] custom errors for ERC-721 tokens.
 */
interface IERC721Errors {
    /**
     * @dev Indicates that an address can't be an owner. For example, `address(0)` is a forbidden owner in ERC-20.
     * Used in balance queries.
     * @param owner Address of the current owner of a token.
     */
    error ERC721InvalidOwner(address owner);

    /**
     * @dev Indicates a `tokenId` whose `owner` is the zero address.
     * @param tokenId Identifier number of a token.
     */
    error ERC721NonexistentToken(uint256 tokenId);

    /**
     * @dev Indicates an error related to the ownership over a particular token. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     * @param tokenId Identifier number of a token.
     * @param owner Address of the current owner of a token.
     */
    error ERC721IncorrectOwner(address sender, uint256 tokenId, address owner);

    /**
     * @dev Indicates a failure with the token `sender`. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     */
    error ERC721InvalidSender(address sender);

    /**
     * @dev Indicates a failure with the token `receiver`. Used in transfers.
     * @param receiver Address to which tokens are being transferred.
     */
    error ERC721InvalidReceiver(address receiver);

    /**
     * @dev Indicates a failure with the `operator`’s approval. Used in transfers.
     * @param operator Address that may be allowed to operate on tokens without being their owner.
     * @param tokenId Identifier number of a token.
     */
    error ERC721InsufficientApproval(address operator, uint256 tokenId);

    /**
     * @dev Indicates a failure with the `approver` of a token to be approved. Used in approvals.
     * @param approver Address initiating an approval operation.
     */
    error ERC721InvalidApprover(address approver);

    /**
     * @dev Indicates a failure with the `operator` to be approved. Used in approvals.
     * @param operator Address that may be allowed to operate on tokens without being their owner.
     */
    error ERC721InvalidOperator(address operator);
}

/**
 * @dev Standard ERC-1155 Errors
 * Interface of the https://eips.ethereum.org/EIPS/eip-6093[ERC-6093] custom errors for ERC-1155 tokens.
 */
interface IERC1155Errors {
    /**
     * @dev Indicates an error related to the current `balance` of a `sender`. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     * @param balance Current balance for the interacting account.
     * @param needed Minimum amount required to perform a transfer.
     * @param tokenId Identifier number of a token.
     */
    error ERC1155InsufficientBalance(address sender, uint256 balance, uint256 needed, uint256 tokenId);

    /**
     * @dev Indicates a failure with the token `sender`. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     */
    error ERC1155InvalidSender(address sender);

    /**
     * @dev Indicates a failure with the token `receiver`. Used in transfers.
     * @param receiver Address to which tokens are being transferred.
     */
    error ERC1155InvalidReceiver(address receiver);

    /**
     * @dev Indicates a failure with the `operator`’s approval. Used in transfers.
     * @param operator Address that may be allowed to operate on tokens without being their owner.
     * @param owner Address of the current owner of a token.
     */
    error ERC1155MissingApprovalForAll(address operator, address owner);

    /**
     * @dev Indicates a failure with the `approver` of a token to be approved. Used in approvals.
     * @param approver Address initiating an approval operation.
     */
    error ERC1155InvalidApprover(address approver);

    /**
     * @dev Indicates a failure with the `operator` to be approved. Used in approvals.
     * @param operator Address that may be allowed to operate on tokens without being their owner.
     */
    error ERC1155InvalidOperator(address operator);

    /**
     * @dev Indicates an array length mismatch between ids and values in a safeBatchTransferFrom operation.
     * Used in batch transfers.
     * @param idsLength Length of the array of token identifiers
     * @param valuesLength Length of the array of token amounts
     */
    error ERC1155InvalidArrayLength(uint256 idsLength, uint256 valuesLength);
}
