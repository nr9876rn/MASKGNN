// ===== FILE: src/ethereum/EthArchive2.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import  "../types/response.sol";
import  "../types/request.sol";
import "../common/IFraxSwapV2Reader.sol";
import {IDoDoV2Reader} from "../common/IDoDoV2Reader.sol";
import {IBancorV2Reader} from "../common/IBancorV2Reader.sol";
import {IBancorV3Reader} from "../common/IBancorV3Reader.sol";
import {IFluidDexLiteReader} from "../common/IFluidDexLiteReader.sol";
import {ILitePsmReader} from "../common/ILitePsmReader.sol";
import {IMaverickV2Reader} from "../common/IMaverickV2Reader.sol";
import {IMooniswapReader} from "../common/IMooniswapReader.sol";
import {IRingSwapReader} from "../common/IRingSwapReader.sol";
import {IMaverickV2Reader} from "../common/IMaverickV2Reader.sol";
import {IAlgebraIntegralReader} from "../common/IAlgebraIntegralReader.sol";
import {IUniswapV3LikeReader} from "../common/IUniswapV3LikeReader.sol";
import {IAaveV3Reader} from "../common/IAaveV3Reader.sol";
import {IUniswapV1Reader} from "../common/IUniswapV1Reader.sol"; // 这里有这个引入就够了

contract EthArchive2 {
    address public immutable ethArchive = 0xB8b9658A406F4EaBD4348030924ae605f3e5831d;
    IDoDoV2Reader public constant doDoV2Reader = IDoDoV2Reader(0x41e248170c8742eF3971ecCBFd5f0ce141f7B898);
    IBancorV2Reader public constant bancorV2Reader = IBancorV2Reader(0x846d71F9Bf3387C9252415A08114A079807647CC);
    IBancorV3Reader public constant bancorV3Reader = IBancorV3Reader(0x38BD1643eed5179c4A805Ec70eF6A2Aaf9Ac602F);
    ILitePsmReader public constant litePsmReader = ILitePsmReader(0x37E7C28f42D93DE7B50d9d5514f6D670Aca490bE);
    IMooniswapReader public constant mooniswapReader = IMooniswapReader(0x4b3ce05Fc839E39Bf67Ff31B17f50eb575f9fCC2);
    IFluidDexLiteReader public constant fluidDexLiteReader = IFluidDexLiteReader(0xf95A47cCBA5F8856Af4666d7eBEC7cc49753F05F);
    IRingSwapReader public constant ringswapReader = IRingSwapReader(0xb198B2636a24f75E0bf1b23C92c775dbd227E916);
    IMaverickV2Reader public constant maverickV2Reader = IMaverickV2Reader(0xfc6A8A91172c9567c0979cEd7dCe5f68385B3d43);
    IAlgebraIntegralReader public constant algebraIntegralReader = IAlgebraIntegralReader(0x2187BCcaCCF582c5d96807AF3871251f436f8580);
    IUniswapV3LikeReader public constant uniswapv3Reader = IUniswapV3LikeReader(0xC372C179Cc668Bbd925e62C61C8D3958A0D0A325);
    IFraxSwapV2Reader public constant fraxSwapV2Reader = IFraxSwapV2Reader(0x94c1D7a4C79a5a284F5b8a0b05509e9f3854164e);
    IAaveV3Reader public constant aaveV3Reader = IAaveV3Reader(0x5A0AD9717d8a97fD4bc7fbd3bDC858ec572164b0);
    IUniswapV1Reader public constant uniswapV1Reader = IUniswapV1Reader(0x98eC1fDd9D1D421926C4d8903bfC232D874231e8);

    function version() public pure returns (uint256) {
        return 20;
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

    function dodoV2PoolInfoV2(DoDoV2Request calldata params) external view returns (DoDoV2ResponseV2 memory response) {
        return doDoV2Reader.dodoV2PoolInfoV2(params);
    }

    function bancorV2PoolInfo(BancorV2Request calldata params) external view returns (BancorV2PoolInfoResponse memory) {
        return bancorV2Reader.bancorV2PoolInfo(params);
    }

    function checkConvertIsValid(BancorV2Request calldata params) external view returns (BancorV2IsValidResponse memory) {
        return bancorV2Reader.checkConvertIsValid(params);
    }

    function getBancorV3PoolInfo(BancorV3Request calldata params) external view returns (BancorV3PoolInfoResponse memory) {
        return bancorV3Reader.getBancorV3PoolInfo(params);
    }

    function checkBancorV3IsValid(BancorV3Request calldata params) external view returns (BancorV3IsValidResponse memory) {
        return bancorV3Reader.checkBancorV3IsValid(params);
    }

    function dssLitePsmInfo(address psmAddr) external view returns (LitePsmResponse memory response) {
        return litePsmReader.dssLitePsmInfo(psmAddr);
    }

    function daiUsdsInfo(address converterAddr) external view returns (DaiUsdsResponse memory response) {
        return litePsmReader.daiUsdsInfo(converterAddr);
    }

    function usddPsmInfo(address psmAddr) external view returns (UsddPsmResponse memory response) {
        return litePsmReader.usddPsmInfo(psmAddr);
    }

    function getFluidDexLitePoolInfo(FluidDexLitePoolRequest calldata params) external returns (FluidDexLiteResponse memory) {
        return fluidDexLiteReader.fluidDexLitePoolInfo(params);
    }

    function isFluidDexLiteDexKeyExist(FluidDexLiteDexIdRequest calldata params) external returns (FluidDexLiteDexIdExistResponse memory) {
        bool exists = fluidDexLiteReader.fluidDexLiteDexKeyExist(params);
        return FluidDexLiteDexIdExistResponse({isExist: exists});
    }

    function fwTokenInfo(address fwToken, address fwFactory) external view returns (FwTokenInfo memory) {
        return ringswapReader.fwTokenInfo(fwToken, fwFactory);
    }

    function maverickV2PoolInfo(MaverickV2Request calldata params) external view returns (MaverickV2Response memory) {
        return maverickV2Reader.maverickV2PoolInfo(params);
    }

    function maverickV2ExtraTicks(MaverickV2ExtraTicksRequest calldata params) external view returns (MaverickV2ExtraTicksResponse memory) {
        return maverickV2Reader.maverickV2ExtraTicks(params);
    }

    function maverickV2Factory(address poolAddr) external view returns (address) {
        return maverickV2Reader.maverickV2Factory(poolAddr);
    }

    function mooniswapPoolInfo(address poolAddr) external view returns (MooniswapResponse memory response) {
        return mooniswapReader.mooniswapPoolInfo(poolAddr);
    }

    function cypherV4Factory(address poolAddr) external view returns (address factoryAddr, address deployer) {
        return algebraIntegralReader.cypherV4Factory(poolAddr);
    }

    function cypherPluginFactory(address pluginAddr) external view returns (address poolAddr, address factoryAddr, address deployer) {
        return algebraIntegralReader.cypherPluginFactory(pluginAddr);
    }

    function cypherV4PoolInfo(AlgebraIntegralLikeRequest calldata request) external view returns (CypherV4Response memory response) {
        return algebraIntegralReader.cypherV4PoolInfo(request);
    }


    function supernovaAlgebraIntegralFactory(address poolAddr) external view returns (address factoryAddr, address deployer) {
        return algebraIntegralReader.supernovaAlgebraIntegralFactory(poolAddr);
    }

    function supernovaPluginFactory(address pluginAddr) external view returns (address poolAddr, address factoryAddr, address deployer) {
        return algebraIntegralReader.supernovaPluginFactory(pluginAddr);
    }

    function supernovaPoolInfo(SupernovaPoolInfoRequest calldata params) external view returns (ThenaIntegralResponse memory) {
        return algebraIntegralReader.supernovaPoolInfo(params);
    }

    function fraxSwapV2PoolInfo(address poolAddr,uint64 count) external view returns (FraxSwapV2Response memory response) {
        return fraxSwapV2Reader.fraxSwapV2PoolInfo(poolAddr, count);
    }

    function solidlyV3PoolInfo(UniswapV3LikeRequest calldata params) public view returns (UniswapV3LikeResponse memory) {
        UniswapV3LikeRequest memory request = params;
        request.dex = Dex.SOLIDLY_V3;
        return uniswapv3Reader.uniswapV3LikePoolInfo(request);
    }

    function solidlyV3Factory(address poolAddr) external view returns (address) {
        return uniswapv3Reader.solidlyV3Factory(poolAddr);
    }

    function solidlyV3LikeTicks(UniswapV3LikeTicksRequest calldata params) public view returns (UniswapV3LikeTicksResponse memory) {
        UniswapV3LikeTicksRequest memory request = params;
        request.dex = Dex.SOLIDLY_V3;
        return uniswapv3Reader.uniswapV3LikeTicks(request);
    }

    function getAaveV3Reserve(address addressesProvider, address underlying) external view returns (AaveV3PoolsResponse memory) {
        return aaveV3Reader.getAaveV3Reserve(addressesProvider, underlying);
    }

    function uniswapV1Factory(address poolAddr) external view returns (address) {
        return uniswapV1Reader.uniswapV1Factory(poolAddr);
    }

    function uniswapV1PoolInfo(address pool) external view returns (UniswapV2LikeResponse memory) {
        return uniswapV1Reader.uniswapV1PoolInfo(pool);
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

// ===== FILE: src/common/IFraxSwapV2Reader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.4;

import {UniswapV2LikeRequest} from "../types/request.sol";
import {UniswapV2LikeResponse, CamelotV2Response, FraxSwapV2Response} from "../types/response.sol";

interface IFraxSwapV2Reader {
   function fraxSwapV2PoolInfo(address pool, uint64 maxSalesRateEndingCount) external view returns (FraxSwapV2Response memory response);
}

// ===== FILE: src/common/IDoDoV2Reader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;
 
import {DoDoV2Request} from "../types/request.sol";
import {DoDoV2ResponseV2} from "../types/response.sol";

interface IDoDoV2Reader {
   function dodoV2PoolInfoV2(DoDoV2Request calldata params) external view returns (DoDoV2ResponseV2 memory response);
}

// ===== FILE: src/common/IBancorV2Reader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {BancorV2Request} from "../types/request.sol";
import {BancorV2PoolInfoResponse, BancorV2IsValidResponse} from "../types/response.sol";

interface IBancorV2Reader {
    /**
     * @notice 获取 Bancor V2.1 池子信息（50/50 权重池）
     * @param params BancorV2Request 请求参数
     * @return response 池子的完整信息
     */
    function bancorV2PoolInfo(BancorV2Request calldata params) external view returns (BancorV2PoolInfoResponse memory response);

    /**
     * @notice 判断 Bancor V2.1 的 converter 是否是有效的（50/50 权重）
     * @param params BancorV2Request 请求参数
     * @return response 是否有效（50/50 权重的池子）
     */
    function checkConvertIsValid(BancorV2Request calldata params) external view returns (BancorV2IsValidResponse memory response);
}



// ===== FILE: src/common/IBancorV3Reader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {BancorV3Request} from "../types/request.sol";
import {BancorV3PoolInfoResponse, BancorV3IsValidResponse} from "../types/response.sol";

interface IBancorV3Reader {
    function getBancorV3PoolInfo(BancorV3Request calldata params) external view returns (BancorV3PoolInfoResponse memory response);
    function checkBancorV3IsValid(BancorV3Request calldata params) external view returns (BancorV3IsValidResponse memory response);
}


// ===== FILE: src/common/IFluidDexLiteReader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {FluidDexLiteDexIdRequest, FluidDexLitePoolRequest} from "../types/request.sol";
import {FluidDexLiteResponse} from "../types/response.sol";

interface IFluidDexLiteReader {
    function fluidDexLiteDexKeyExist(FluidDexLiteDexIdRequest calldata params) external returns (bool isDexIdExist);
    function fluidDexLitePoolInfo(FluidDexLitePoolRequest calldata params) external returns (FluidDexLiteResponse memory response);
}


// ===== FILE: src/common/ILitePsmReader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.4;

import {LitePsmResponse, DaiUsdsResponse, UsddPsmResponse} from "../types/response.sol";

interface ILitePsmReader {
    function dssLitePsmInfo(address psmAddr) external view returns (LitePsmResponse memory response);
    function daiUsdsInfo(address converterAddr) external view returns (DaiUsdsResponse memory response);
    function usddPsmInfo(address psmAddr) external view returns (UsddPsmResponse memory response);
}

// ===== FILE: src/common/IMaverickV2Reader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {MaverickV2Request, MaverickV2ExtraTicksRequest} from "../types/request.sol";
import {MaverickV2Response, MaverickV2ExtraTicksResponse} from "../types/response.sol";

interface IMaverickV2Reader {
    function maverickV2PoolInfo(MaverickV2Request calldata params) external view returns (MaverickV2Response memory response);
    function maverickV2ExtraTicks(MaverickV2ExtraTicksRequest calldata params) external view returns (MaverickV2ExtraTicksResponse memory response);
    function maverickV2Factory(address poolAddr) external view returns (address);
}


// ===== FILE: src/common/IMooniswapReader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.4;

import {MooniswapResponse} from "../types/response.sol";

interface IMooniswapReader {
    function mooniswapFactory(address poolAddr) external view returns (address);
    function mooniswapPoolInfo(address poolAddr) external view returns (MooniswapResponse memory response);
}

// ===== FILE: src/common/IRingSwapReader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {RingSwapV2Response, FwTokenInfo} from "../types/response.sol";

interface IRingSwapReader {
    // 方法需要去掉view，已经测试过对上下游无影响,不会破坏交易流程或额外gas估算，不会破坏聚合器读接口稳定性。
    function ringSwapV2PoolInfo(address poolAddr) external  returns (RingSwapV2Response memory response);
    function fwTokenInfo(address token, address fwFactory) external view returns (FwTokenInfo memory info);
}



// ===== FILE: src/common/IAlgebraIntegralReader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {Dex, AlgebraIntegralLikeRequest, SupernovaPoolInfoRequest, WDexRequest} from "../types/request.sol";
import {ThenaIntegralResponse, AlgebraIntegralBasePool, QuickswapV4Response, CamelotV4Response,CypherV4Response, WDexResponse} from "../types/response.sol";

interface IAlgebraIntegralReader {
    function algebraIntegralFactory(Dex dex, address poolAddr) external view returns (address, address);
    function thenaIntegralFactory(address poolAddr) external view returns (address factoryAddr, address deployer);
    function quickswapV4Factory(address poolAddr) external view returns (address factoryAddr, address deployer);
    function trebleswapFactory(address poolAddr) external view returns (address factoryAddr, address deployer);
    function trebleswapV2Factory(address poolAddr) external view returns (address factoryAddr, address deployer);
    function cypherV4Factory(address poolAddr) external view returns (address factoryAddr, address deployer);
    function hydrexV4Factory(address poolAddr) external view returns (address factoryAddr, address deployer);
    function camelotV4Factory(address poolAddr) external view returns (address factoryAddr, address deployer);
    function wDexFactory(address poolAddr) external view returns (address factoryAddr, address deployer);

    function basePluginFactory(Dex dex, address pluginAddr) external view returns (address poolAddr, address factoryAddr, address deployer);
    function thenaBasePluginFactory(Dex dex, address pluginAddr) external view returns (address, address, address);
    function thenaPluginFactory(address pluginAddr) external view returns (address poolAddr, address factoryAddr, address deployer);
    function camelotPluginFactory(address pluginAddr) external view returns (address poolAddr, address factoryAddr, address deployer);
    function hydrexPluginFactory(address pluginAddr) external view returns (address poolAddr, address factoryAddr, address deployer);
    function cypherPluginFactory(address pluginAddr) external view returns (address poolAddr, address factoryAddr, address deployer);
    function supernovaPluginFactory(address pluginAddr) external view returns (address poolAddr, address factoryAddr, address deployer);
    function wDexPluginFactory(address pluginAddr) external view returns (address poolAddr, address factoryAddr, address deployer);

    function algebraIntegralBasePoolInfo(AlgebraIntegralLikeRequest calldata params) external view returns (AlgebraIntegralBasePool memory);
    function thenaIntegralPoolInfo(AlgebraIntegralLikeRequest calldata params) external view returns (ThenaIntegralResponse memory);
    function quickswapV4PoolInfo(AlgebraIntegralLikeRequest calldata params) external view returns (QuickswapV4Response memory);
    function trebleswapPoolInfo(AlgebraIntegralLikeRequest calldata params) external view returns (QuickswapV4Response memory);
    function trebleswapV2PoolInfo(AlgebraIntegralLikeRequest calldata params) external view returns (QuickswapV4Response memory);
    function camelotV4PoolInfo(AlgebraIntegralLikeRequest calldata params) external view returns (CamelotV4Response memory);
    function supernovaAlgebraIntegralFactory(address poolAddr) external view returns (address factoryAddr, address deployer);
    function supernovaPoolInfo(SupernovaPoolInfoRequest calldata params) external view returns (ThenaIntegralResponse memory);
    function wDexPoolInfo(WDexRequest calldata params) external view returns (WDexResponse memory);
    function hydrexV4PoolInfo(AlgebraIntegralLikeRequest calldata params) external view returns (CamelotV4Response memory);
    function cypherV4PoolInfo(AlgebraIntegralLikeRequest calldata params) external view returns (CypherV4Response memory);

   function nestCLPoolInfo(AlgebraIntegralLikeRequest calldata params) external view returns (CamelotV4Response memory response);

}


// ===== FILE: src/common/IUniswapV3LikeReader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {UniswapV3LikeRequest, UniswapV3LikeTicksRequest} from "../types/request.sol";
import {UniswapV3LikeResponse, UniswapV3LikeTicksResponse} from "../types/response.sol";

interface IUniswapV3LikeReader {
   function uniswapV3LikeFactory(address poolAddr) external view returns (address);
   function squadswapV3LikeFactory(address poolAddr) external view returns (address);
   function uniswapV3LikePoolInfo(UniswapV3LikeRequest calldata params) external view returns (UniswapV3LikeResponse memory response);
   function uniswapV3LikeTicks(UniswapV3LikeTicksRequest calldata params) external view returns (UniswapV3LikeTicksResponse memory response);
   function solidlyV3Factory(address poolAddr) external view returns (address);
}

// ===== FILE: src/common/IAaveV3Reader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {AaveV3PoolsResponse} from "../types/response.sol";

interface IAaveV3Reader {
    function getAaveV3Reserve(address addressesProvider, address underlying)
        external
        view
        returns (AaveV3PoolsResponse memory response);
}


// ===== FILE: src/common/IUniswapV1Reader.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.4;

import {UniswapV2LikeResponse} from "../types/response.sol";

interface IUniswapV1Reader {
    function uniswapV1Factory(address poolAddr) external view returns (address);
    function uniswapV1PoolInfo(address pool) external view returns (UniswapV2LikeResponse memory response);
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