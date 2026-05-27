// ===== FILE: src/vault/CoreVault.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { IERC20 } from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import { IERC20Metadata } from "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import { SafeERC20 } from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import { Pausable } from "@openzeppelin/contracts/utils/Pausable.sol";
import { ReentrancyGuard } from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import { IERC165 } from "@openzeppelin/contracts/utils/introspection/IERC165.sol";
import { CoreAccess } from "../base/CoreAccess.sol";
import { IAssetCustodian } from "../interfaces/IAssetCustodian.sol";
import { IBlocklist } from "../interfaces/IBlocklist.sol";
import { IFeeManager } from "../interfaces/IFeeManager.sol";
import { IERC7575 } from "../interfaces/IERC7575.sol";
import { INAVOracle } from "../interfaces/INAVOracle.sol";
import { IRebalanceManager } from "../interfaces/IRebalanceManager.sol";
import { IRequestManager } from "../interfaces/IRequestManager.sol";
import { IShareToken } from "../interfaces/IShareToken.sol";
import { IWhitelist } from "../interfaces/IWhitelist.sol";
import { IVaultSettlement } from "../interfaces/IVaultSettlement.sol";
import {
    BatchHint,
    OndoTarget,
    PassiveRebalanceParams,
    RebalanceBuy,
    RebalanceTrigger,
    SignedPriceBundle,
    Trade
} from "../interfaces/Types.sol";
import { FixedPoint } from "../libraries/FixedPoint.sol";
import { Roles } from "../libraries/Roles.sol";

/// @title CoreVault
contract CoreVault is CoreAccess, Pausable, ReentrancyGuard, IVaultSettlement, IERC7575 {
    using SafeERC20 for IERC20;

    uint256 internal constant MAX_STOCK_TOKENS = 20;

    enum VaultStatus {
        Normal,
        Rebalancing,
        Paused
    }

    IShareToken public shareToken;
    IERC20 public usdc;
    INAVOracle public navOracle;
    IWhitelist public whitelist;
    IBlocklist public blocklist;
    IFeeManager public feeManager;
    IRequestManager public requestManager;
    IRebalanceManager public rebalanceManager;
    IAssetCustodian public ac;
    VaultStatus public vaultStatus;
    uint256 public minRedeemAssets;
    uint16 public defaultSlippageBps;
    uint16 public defaultLossPercentBps;

    uint256 public minDeposit;
    uint256 public maxDepositAmount;
    uint256 public tvlCap;
    uint256 public completedDepositAssets;
    uint256 public pendingDepositAssets;

    uint256 public latestTotalAssetsValue;
    mapping(uint256 => uint256) public totalAssetsByRound;

    address[] public _stockTokens;
    mapping(address => uint256) public targetWeightBps;

    bool public whitelistRequired;

    string public vaultName;
    string public vaultSymbol;

    /// @notice Subscription fee rate, in bps (per ten thousand), for example, 50 = 0.5%.
    uint16 public depositFeeBps;
    address public feeRecipient;

    bool private _coreVaultInitialized;

    event ACSet(address indexed ac);
    event FeeManagerSet(address indexed feeManager);
    event RequestManagerSet(address indexed requestManager);
    event RebalanceManagerSet(address indexed rebalanceManager);
    event Deposit(address indexed sender, address indexed owner, uint256 assets, uint256 shares);
    event Withdraw(
        address indexed sender, address indexed receiver, address indexed owner, uint256 assets, uint256 shares
    );
    event DepositRequested(
        uint256 indexed reqId, address indexed owner, uint256 assets, uint256 shares, bytes32 groupId
    );
    event RedeemRequested(uint256 indexed reqId, address indexed owner, uint256 shares, bytes32 groupId);
    event DepositFulfilled(uint256 indexed reqId, address indexed owner, uint256 shares, uint256 refundUsdc);
    event RedeemFulfilled(uint256 indexed reqId, address indexed owner, uint256 shares, uint256 usdcReceived);
    event RequestFailed(uint256 indexed reqId, bool deposit, bytes reason);
    event OndoTargetsSet(bytes32 indexed allocationHash);
    event RebalanceRequested(bytes32 indexed rbId, bytes32 indexed groupId);
    event RebalanceFailed(bytes32 indexed rbId, bytes reason);
    event PassiveRebalanceRequested(
        bytes32 indexed rbId,
        bytes32 indexed groupId,
        RebalanceTrigger indexed trigger,
        uint8 riskLevel,
        bytes32 reasonHash
    );
    event RebalanceFulfilled(bytes32 indexed rbId, uint256 totalAssetsAfter);
    event TotalAssetsUpdated(uint256 indexed round, uint256 totalAssetsValue);
    event VaultForcedNormal(address indexed caller);

    modifier onlyAc() {
        _onlyAc();
        _;
    }

    modifier onlyNormalVault() {
        _onlyNormalVault();
        _;
    }

    function supportsInterface(bytes4 interfaceId) external pure returns (bool) {
        return interfaceId == type(IERC165).interfaceId || interfaceId == type(IERC7575).interfaceId;
    }

    function share() external view returns (address shareTokenAddress) {
        return address(shareToken);
    }

    function asset() external view returns (address assetTokenAddress) {
        return address(usdc);
    }

    constructor(
        address admin,
        string memory vaultName_,
        string memory vaultSymbol_,
        address shareToken_,
        address usdc_,
        address navOracle_,
        address whitelist_,
        address blocklist_
    ) CoreAccess(admin) {
        if (admin != address(0)) {
            _initializeCoreVault(vaultName_, vaultSymbol_, shareToken_, usdc_, navOracle_, whitelist_, blocklist_);
        }
    }

    function initialize(
        address admin,
        string memory vaultName_,
        string memory vaultSymbol_,
        address shareToken_,
        address usdc_,
        address navOracle_,
        address whitelist_,
        address blocklist_
    ) external {
        _initializeCoreAccess(admin);
        _initializeCoreVault(vaultName_, vaultSymbol_, shareToken_, usdc_, navOracle_, whitelist_, blocklist_);
    }

    function _initializeCoreVault(
        string memory vaultName_,
        string memory vaultSymbol_,
        address shareToken_,
        address usdc_,
        address navOracle_,
        address whitelist_,
        address blocklist_
    ) internal {
        require(!_coreVaultInitialized, "CV:INIT");
        require(shareToken_ != address(0) && usdc_ != address(0) && navOracle_ != address(0), "CV:ZA");
        _coreVaultInitialized = true;
        vaultName = vaultName_;
        vaultSymbol = vaultSymbol_;
        shareToken = IShareToken(shareToken_);
        usdc = IERC20(usdc_);
        navOracle = INAVOracle(navOracle_);
        whitelist = IWhitelist(whitelist_);
        blocklist = IBlocklist(blocklist_);
    }

    function setAc(address ac_) external onlyAdmin {
        require(ac_ != address(0), "CV:ZA");
        ac = IAssetCustodian(ac_);
        emit ACSet(ac_);
    }

    function setFeeManager(address feeManager_) external onlyAdmin {
        feeManager = IFeeManager(feeManager_);
        emit FeeManagerSet(feeManager_);
    }

    function setRequestManager(address requestManager_) external onlyAdmin {
        require(requestManager_ != address(0), "CV:ZA");
        requestManager = IRequestManager(requestManager_);
        emit RequestManagerSet(requestManager_);
    }

    function setRebalanceManager(address rebalanceManager_) external onlyAdmin {
        require(rebalanceManager_ != address(0), "CV:ZA");
        rebalanceManager = IRebalanceManager(rebalanceManager_);
        emit RebalanceManagerSet(rebalanceManager_);
    }

    function setRiskParams(uint256 minDeposit_, uint256 maxDeposit_, uint256 tvlCap_, uint256 minRedeemAssets_) external onlyAdmin {
        require(minDeposit_ <= maxDeposit_, "CV:DR");
        minDeposit = minDeposit_;
        maxDepositAmount = maxDeposit_;
        minRedeemAssets = minRedeemAssets_;
        tvlCap = tvlCap_;
    }

    function setDefaultSlippageBps(uint16 slippageBps) external onlyAdmin {
        require(slippageBps <= 2000, "CV:SL");
        defaultSlippageBps = slippageBps;
    }

    function setDefaultLossPercentBps(uint16 lossPercentBps) external onlyAdmin {
        require(lossPercentBps <= 10_000, "CV:LP");
        defaultLossPercentBps = lossPercentBps;
    }

    function setWhitelistRequired(bool required) external onlyAdmin {
        whitelistRequired = required;
    }

    function setFeeParams(uint16 depositFeeBps_, address feeRecipient_) external onlyAdmin {
        require(feeRecipient_ != address(0), "CV:ZA");
        depositFeeBps = depositFeeBps_;
        feeRecipient = feeRecipient_;
    }

    function setOndoTargets(OndoTarget[] calldata targets) external onlyRole(Roles.ONDO_TARGET_MANAGER) {
        _validateOndoTargets(targets);
        for (uint256 i; i < _stockTokens.length; ++i) {
            targetWeightBps[_stockTokens[i]] = 0;
        }
        delete _stockTokens;
        for (uint256 i; i < targets.length; ++i) {
            _stockTokens.push(targets[i].token);
            targetWeightBps[targets[i].token] = targets[i].bps;
        }
        emit OndoTargetsSet(_ondoAllocationHash());
    }

    function pause() external onlyRole(Roles.PAUSE_ROLE) {
        vaultStatus = VaultStatus.Paused;
        _pause();
    }

    function unpause() external onlyRole(Roles.PAUSE_ROLE) {
        _unpause();
        if (address(rebalanceManager) != address(0) && rebalanceManager.activeRebalanceId() != bytes32(0)) {
            vaultStatus = VaultStatus.Rebalancing;
        } else {
            vaultStatus = VaultStatus.Normal;
        }
    }

    function forceNormal() external onlyAdmin {
        if (paused()) _unpause();
        vaultStatus = VaultStatus.Normal;
        emit VaultForcedNormal(msg.sender);
    }

    /// @notice User initiates USDC purchase.
    /// @dev Process: Verify price bundle signature -> Calculate expected Share -> Retrieve user's USDC -> Transfer to AC -> AC calls Proxy.
    /// @param assets Quantity of USDC purchased, 6-digit precision.
    /// @param owner Source of funds and final Share recipient.
    /// @param priceBundle Price bundle signed by the trusted signer.
    /// @param minSharesOut Minimum acceptable shares for slippage protection.
    /// @return reqId ID of this asynchronous request.
    /// @return feeAmount USDC subscription fee charged for this request.
    function requestDeposit(
        uint256 assets,
        address owner,
        SignedPriceBundle calldata priceBundle,
        uint256 minSharesOut
    )
        external
        whenNotPaused
        onlyNormalVault
        nonReentrant
        returns (uint256 reqId, uint256 feeAmount)
    {
        uint64 round = navOracle.ingest(priceBundle);
        _updateTotalAssetsForRound(round);
        _accrueManagementFeeIfConfigured();
        return _requestDeposit(assets, owner, round, minSharesOut);
    }

    /// @notice User initiates a share redemption.
    // @dev Process: Verify price bundle signature -> Lock user Share -> Transfer underlying assets to AC
    // according to share -> AC calls Proxy Sell.
    // @param shares Share token amount to redeem, 18-digit precision.
    // @param owner Source of Shares and final USDC recipient.
    // @param priceBundle Price bundle signed by the trusted signer.
    // @return reqId ID of this asynchronous request.
    function requestRedeem(uint256 shares, address owner, SignedPriceBundle calldata priceBundle)
        external
        whenNotPaused
        onlyNormalVault
        nonReentrant
        returns (uint256 reqId)
    {
        uint64 round = navOracle.ingest(priceBundle);
        _updateTotalAssetsForRound(round);
        _accrueManagementFeeIfConfigured();
        return _requestRedeem(shares, owner, round);
    }


    /// @notice Curator active rebalancing entry point. Uses the same two-phase sell/buy flow as passive rebalancing.
    function requestRebalance(
        Trade[] calldata sells,
        RebalanceBuy[] calldata buys,
        address[] calldata targetTokens,
        uint256[] calldata targetWeights,
        SignedPriceBundle calldata priceBundle
    )
        external
        onlyRole(Roles.CURATOR)
        whenNotPaused
        onlyNormalVault
        nonReentrant
        returns (bytes32 rbId, bytes32 groupId)
    {
        require(address(rebalanceManager) != address(0), "CV:RB0");
        uint64 round = navOracle.ingest(priceBundle);
        _updateTotalAssetsForRound(round);
        _accrueManagementFeeIfConfigured();
        rbId = rebalanceManager.prepareActiveRebalance(sells.length, buys.length, buys, targetTokens, targetWeights);
        _savePendingOndoTargets(rbId, targetTokens, targetWeights);
        vaultStatus = VaultStatus.Rebalancing;
        groupId = _dispatchPassiveRebalance(rbId, sells, buys, RebalanceTrigger.Active);
        require(groupId != bytes32(0), "CV:ZB");
        emit RebalanceRequested(rbId, groupId);
    }

    /// @notice Monitor Service Passive portfolio rebalancing entry point.
    function requestPassiveRebalance(PassiveRebalanceParams calldata params)
        external
        onlyRole(Roles.MONITOR)
        whenNotPaused
        onlyNormalVault
        nonReentrant
        returns (bytes32 rbId, bytes32 groupId)
    {
        require(address(rebalanceManager) != address(0), "CV:RB0");
        uint64 round = navOracle.ingest(params.priceBundle);
        _updateTotalAssetsForRound(round);
        _accrueManagementFeeIfConfigured();
        rbId = rebalanceManager.preparePassiveRebalance(
            params.sells.length,
            params.buys.length,
            params.trigger,
            params.riskLevel,
            params.reasonHash,
            params.buys,
            params.targetTokens,
            params.targetWeights
        );
        _savePendingOndoTargets(rbId, params.targetTokens, params.targetWeights);
        vaultStatus = VaultStatus.Rebalancing;
        // Distribute and adjust stacks to reduce the stack depth of individual functions.
        groupId = _dispatchPassiveRebalance(rbId, params.sells, params.buys, params.trigger);
        require(groupId != bytes32(0), "CV:ZB");
        emit PassiveRebalanceRequested(rbId, groupId, params.trigger, params.riskLevel, params.reasonHash);
    }

    /// --------------------------------- Settlement Entry Points (Called by AC) -----------------------------------
    function settleDeposit(
        bytes32 groupId,
        uint256 reqId,
        address[] calldata etfTokens,
        uint256[] calldata amounts,
        uint256 refundUsdc,
        uint256 navRound
    ) external onlyAc nonReentrant {
        require(etfTokens.length == amounts.length, "CV:LEN");
        _validateSettlementNavRound(navRound);
        IRequestManager.DepositSettlement memory result =
            requestManager.settleDeposit(groupId, reqId, refundUsdc);
        pendingDepositAssets -= result.spent + result.refund;
        completedDepositAssets += result.spent;
        if (result.shares > 0) {
            shareToken.mint(result.owner, result.shares);
            if (address(feeManager) != address(0)) feeManager.startAccrual();
            emit DepositFulfilled(reqId, result.owner, result.shares, 0);
        }
    }

    function settleRedeem(bytes32 groupId, uint256 reqId, uint256 usdcReceived, uint256 navRound)
        external
        onlyAc
        nonReentrant
    {
        _validateSettlementNavRound(navRound);
        IRequestManager.RedeemSettlement memory result =
            requestManager.settleRedeem(groupId, reqId, usdcReceived);
        if (result.fulfilledShares > 0) {
            shareToken.burn(address(this), result.fulfilledShares);
        }
        if (result.payout > 0) {
            usdc.safeTransfer(result.owner, result.payout);
        }
        emit RedeemFulfilled(reqId, result.owner, result.fulfilledShares, result.payout);
    }

    function settleRebalance(bytes32 rbId, address[] calldata tokens, uint256[] calldata amounts)
        external
        onlyAc
        nonReentrant
    {
        require(tokens.length == amounts.length, "CV:LEN");
        uint256 round = navOracle.latestRound();
        uint256 totalAssetsAfter = _updateTotalAssetsForRound(round);
        vaultStatus = paused() ? VaultStatus.Paused : VaultStatus.Normal;
        rebalanceManager.settleRebalanceComplete(rbId, tokens, amounts);
        _applyPendingOndoTargets(rbId);
        emit RebalanceFulfilled(rbId, totalAssetsAfter);
    }

    function settleRebalanceFailure(bytes32 rbId, bytes calldata reason) external onlyAc nonReentrant {
        vaultStatus = paused() ? VaultStatus.Paused : VaultStatus.Normal;
        rebalanceManager.settleRebalanceFailed(rbId);
        _clearPendingOndoTargets(rbId);
        emit RebalanceFailed(rbId, reason);
    }

    function settleFailure(bytes32 groupId, uint256 reqId, uint256 returnedUsdc, bytes calldata reason)
        external
        onlyAc
        nonReentrant
    {
        IRequestManager.RequestData memory request = requestManager.getRequest(reqId);
        IRequestManager.FailureSettlement memory result =
            requestManager.settleRequestFailure(groupId, reqId, returnedUsdc);
        if (result.isDeposit) {
            pendingDepositAssets -= request.assets;
            if (result.refund > 0) usdc.safeTransfer(result.owner, result.refund);
            emit RequestFailed(reqId, true, reason);
        } else {
            IERC20(address(shareToken)).safeTransfer(result.owner, result.shares);
            emit RequestFailed(reqId, false, reason);
        }
    }

    /// ---------------------------Fee Calculate---------------------------------

    /// @notice Settle dividend fees with the latest signed price package
    function settlePerformanceFee(SignedPriceBundle calldata priceBundle)
        external
        onlyRole(Roles.OPERATOR)
        nonReentrant
        returns (uint256)
    {
        require(vaultStatus == VaultStatus.Normal, "CV:NN");
        uint64 round = navOracle.ingest(priceBundle);
        _updateTotalAssetsForRound(round);
        _accrueManagementFeeIfConfigured();
        if (address(feeManager) == address(0)) return 0;
        uint256 totalAssetsValue = _totalAssetsForRound(round);
        (uint256 shares,) = feeManager.settlePerformanceFee(totalAssetsValue, effectiveTotalSupply());
        return shares;
    }

    /// @dev Lazy loading supplements management fees
    function _accrueManagementFeeIfConfigured() internal returns (uint256 shares) {
        if (address(feeManager) == address(0)) return 0;
        uint256 supply = effectiveTotalSupply();
        if (supply == 0) return 0;
        shares = feeManager.accrueManagementFee(totalAssets(), supply);
    }

    ///  ---------------------------Public Function---------------------------------

    function getTargetAllocation() public view returns (address[] memory tokens, uint256[] memory weights) {
        tokens = new address[](_stockTokens.length);
        weights = new uint256[](_stockTokens.length);
        for (uint256 i; i < _stockTokens.length; ++i) {
            tokens[i] = _stockTokens[i];
            weights[i] = targetWeightBps[_stockTokens[i]];
        }
    }

    function totalAssets() public view returns (uint256) {
        return latestTotalAssetsValue;
    }

    /// @notice After updating the price with the latest signed price package, return the net asset value per share (NAV) with 18 digits precision.
    // @dev NAV = totalAssetsValue / effectiveTotalSupply. The initial NAV is 1e18 (i.e., 1 USDC/share).
    function navPerShare(SignedPriceBundle calldata priceBundle) public returns (uint256) {
        uint64 round = navOracle.ingest(priceBundle);
        uint256 totalAssetsValue = _updateTotalAssetsForRound(round);
        uint256 supply = effectiveTotalSupply() - requestManager.escrowedShares();
        if (supply == 0) return 1e18;
        return totalAssetsValue * 1e18 / supply;
    }

    function totalSupply() public view returns (uint256) {
        return IERC20(address(shareToken)).totalSupply();
    }

    /// @notice Returns the effective total share used for valuation, which is the actual total supply plus the share of uncrystallized costs.
    function effectiveTotalSupply() public view returns (uint256) {
        if (address(feeManager) == address(0)) return totalSupply();
        return feeManager.effectiveTotalSupply(totalSupply());
    }

    function convertToShares(uint256 assets, SignedPriceBundle calldata priceBundle) external view returns (uint256) {
        return _previewDeposit(
            FixedPoint.usdcToWad(assets),
            navOracle.recomputeTotalAssetsFromBundle(address(this), address(usdc), priceBundle),
            effectiveTotalSupply() - requestManager.escrowedShares()
        );
    }

    function previewDepositWithFee(uint256 assets, SignedPriceBundle calldata priceBundle)
        external
        view
        returns (uint256 shares, uint256 feeAmount)
    {
        feeAmount = assets * depositFeeBps / 10_000;
        uint256 netAssets = assets - feeAmount;
        shares = _previewDeposit(
            FixedPoint.usdcToWad(netAssets),
            navOracle.recomputeTotalAssetsFromBundle(address(this), address(usdc), priceBundle),
            effectiveTotalSupply() - requestManager.escrowedShares()
        );
    }

    function convertToAssets(uint256 shares, SignedPriceBundle calldata priceBundle) external view returns (uint256) {
        uint256 supply = effectiveTotalSupply() - requestManager.escrowedShares();
        if (supply == 0) return 0;
        return FixedPoint.wadToUsdc(
            shares * navOracle.recomputeTotalAssetsFromBundle(address(this), address(usdc), priceBundle) / supply
        );
    }

    ///  ---------------------------Private Function---------------------------------

    function _requestDeposit(uint256 assets, address owner, uint64 round, uint256 minSharesOut)
        internal
        returns (uint256 reqId, uint256 feeAmount)
    {
        require(address(ac) != address(0), "CV:AC0");
        require(address(requestManager) != address(0), "CV:RM0");
        require(assets >= minDeposit && assets <= maxDepositAmount, "CV:DB");
        _checkAccount(owner);
        require(msg.sender == owner, "CV:OP");
        // Deposit Fee
        feeAmount = assets * depositFeeBps / 10_000;
        uint256 netAssets = assets - feeAmount;
        require(totalAssets() + FixedPoint.usdcToWad(pendingDepositAssets + netAssets) <= tvlCap, "CV:CAP");
        if (feeAmount > 0 && feeRecipient != address(0)) {
            usdc.safeTransferFrom(owner, feeRecipient, feeAmount);
        }
        usdc.safeTransferFrom(owner, address(ac), netAssets);
        pendingDepositAssets += netAssets;
        // execution
        uint256 totalAssetsValue = _totalAssetsForRound(round);
        uint256 supply = effectiveTotalSupply() - requestManager.escrowedShares();
        uint256 shares = _previewDeposit(FixedPoint.usdcToWad(netAssets), totalAssetsValue, supply);
        require(shares >= minSharesOut, "CV:MS");
        reqId = requestManager.createDepositRequest(owner, netAssets, shares, round, totalAssetsValue, supply);
        bytes32 expectedGroupId = ac.previewNextGroupId();
        requestManager.setGroupId(reqId, expectedGroupId);
        bytes32 groupId = ac.executeDeposit(reqId, netAssets, _defaultBatchHint());
        require(groupId == expectedGroupId, "CV:GB");
        emit Deposit(msg.sender, owner, assets, shares);
        emit DepositRequested(reqId, owner, netAssets, shares, groupId);
    }

    function _requestRedeem(uint256 shares, address owner, uint64 round)
        internal
        returns (uint256 reqId)
    {
        require(address(ac) != address(0), "CV:AC0");
        require(address(requestManager) != address(0), "CV:RM0");
        require(shares > 0, "CV:ZS");
        _checkAccount(owner);
        require(msg.sender == owner, "CV:OP");
        uint256 totalAssetsValue = _totalAssetsForRound(round);
        uint256 availableSupply = effectiveTotalSupply() - requestManager.escrowedShares();
        require(totalAssetsValue > 0 && availableSupply > 0, "CV:ZR");
        require(shares <= availableSupply, "CV:RS");
        uint256 expectedAssets = shares * totalAssetsValue / availableSupply;
        require(minRedeemAssets == 0 || expectedAssets >= FixedPoint.usdcToWad(minRedeemAssets), "CV:BRA");
        reqId = requestManager.createRedeemRequest(owner, shares, round, totalAssetsValue, availableSupply, expectedAssets);
        bytes32 expectedGroupId = ac.previewNextGroupId();
        requestManager.setGroupId(reqId, expectedGroupId);
        IERC20(address(shareToken)).safeTransferFrom(owner, address(this), shares);
        bytes32 groupId = _dispatchRedeem(reqId, shares, availableSupply);
        require(groupId == expectedGroupId, "CV:GB");
        emit Withdraw(msg.sender, owner, owner, 0, shares);
        emit RedeemRequested(reqId, owner, shares, groupId);
    }

    function _dispatchRedeem(uint256 reqId, uint256 shares, uint256 supply)
        internal
        returns (bytes32 groupId)
    {
        (address[] memory tokens, uint256[] memory amounts) = _transferRedeemBasketToAc(shares, supply);
        groupId = ac.executeRedeem(reqId, tokens, amounts, _defaultBatchHint());
    }

    function _dispatchPassiveRebalance(
        bytes32 rbId,
        Trade[] calldata sells,
        RebalanceBuy[] calldata buys,
        RebalanceTrigger trigger
    ) internal returns (bytes32 groupId) {
        _transferRebalanceSellsToAc(sells);
        groupId = ac.executeRebalance(rbId, sells, buys, trigger, _defaultBatchHint());
    }

    function _transferRebalanceSellsToAc(Trade[] calldata sells) internal {
        for (uint256 i; i < sells.length; ++i) {
            require(sells[i].amountIn > 0, "CV:ZS");
            IERC20(sells[i].tokenIn).safeTransfer(address(ac), sells[i].amountIn);
        }
    }

    function _transferRedeemBasketToAc(uint256 shares, uint256 supply)
        internal
        returns (address[] memory tokens, uint256[] memory amounts)
    {
        uint256 count = _stockTokens.length;
        tokens = new address[](count);
        amounts = new uint256[](count);
        if (supply == 0) return (tokens, amounts);
        for (uint256 i; i < count; ++i) {
            address token = _stockTokens[i];
            tokens[i] = token;
            uint256 bal = IERC20(token).balanceOf(address(this));
            uint256 amt = bal * shares / supply;
            amounts[i] = amt;
            if (amt > 0) IERC20(token).safeTransfer(address(ac), amt);
        }
    }

    function _defaultBatchHint() internal view returns (BatchHint memory) {
        return BatchHint({ maxSlippageBps: defaultSlippageBps, maxLossPercentBps: defaultLossPercentBps });
    }


    function _setOndoTargetsFromPending(bytes32 rbId) internal {
        OndoTarget[] memory pending = rebalanceManager.getPendingOndoTargets(rbId);
        for (uint256 i; i < _stockTokens.length; ++i) {
            targetWeightBps[_stockTokens[i]] = 0;
        }
        delete _stockTokens;
        for (uint256 i; i < pending.length; ++i) {
            _stockTokens.push(pending[i].token);
            targetWeightBps[pending[i].token] = pending[i].bps;
        }
        emit OndoTargetsSet(_ondoAllocationHash());
    }

    function _savePendingOndoTargets(
        bytes32 rbId,
        address[] calldata targetTokens,
        uint256[] calldata targetWeights
    ) internal {
        require(targetTokens.length <= MAX_STOCK_TOKENS, "CV:TL");
        rebalanceManager.savePendingOndoTargets(rbId, targetTokens, targetWeights);
    }

    function _applyPendingOndoTargets(bytes32 rbId) internal {
        _setOndoTargetsFromPending(rbId);
        _clearPendingOndoTargets(rbId);
    }

    function _clearPendingOndoTargets(bytes32 rbId) internal {
        rebalanceManager.clearPendingOndoTargets(rbId);
    }

    function _validateOndoTargets(OndoTarget[] calldata targets) internal pure {
        require(targets.length <= MAX_STOCK_TOKENS, "CV:TL");
        uint256 total;
        for (uint256 i; i < targets.length; ++i) {
            require(targets[i].token != address(0), "CV:ZA");
            for (uint256 j; j < i; ++j) {
                require(targets[i].token != targets[j].token, "CV:DUP");
            }
            total += targets[i].bps;
        }
        require(total == 10_000, "CV:BT");
    }

    function _ondoAllocationHash() internal view returns (bytes32) {
        // forge-lint: disable-next-line(asm-keccak256)
        return keccak256(abi.encode(_stockTokens, _targetWeights()));
    }

    function _onlyAc() internal view {
        require(msg.sender == address(ac), "CV:AC");
    }

    function _onlyNormalVault() internal view {
        require(vaultStatus == VaultStatus.Normal, "CV:NN");
    }

    function _checkAccount(address account) internal view {
        require(account != address(0), "CV:ZA");
        require(!blocklist.isBlocked(account), "CV:BLK");
        if (whitelistRequired) require(whitelist.isWhitelisted(account), "CV:WL");
    }

    function _recomputedTotalAssetsFromOraclePrices() internal view returns (uint256 totalAssetsValue) {
        totalAssetsValue = FixedPoint.usdcToWad(usdc.balanceOf(address(this)));
        uint256 count = _stockTokens.length;
        for (uint256 i; i < count; ++i) {
            address token = _stockTokens[i];
            if (token == address(usdc)) continue;
            uint256 balance = IERC20(token).balanceOf(address(this));
            if (balance == 0) continue;
            uint256 price = navOracle.priceOf(token);
            require(price > 0, "CV:MP");
            totalAssetsValue += balance * price / _assetUnit(token);
        }
    }

    function _targetWeights() internal view returns (uint256[] memory weights) {
        weights = new uint256[](_stockTokens.length);
        for (uint256 i; i < _stockTokens.length; ++i) {
            weights[i] = targetWeightBps[_stockTokens[i]];
        }
    }

    function _validateSettlementNavRound(uint256 navRound) internal view {
        require(navRound != 0 && navRound <= navOracle.latestRound(), "CV:INR");
    }

    function _assetUnit(address token) internal view returns (uint256) {
        uint8 decimals = IERC20Metadata(token).decimals();
        require(decimals <= 36, "CV:DEC");
        return 10 ** uint256(decimals);
    }

    function _updateTotalAssetsForRound(uint256 round) internal returns (uint256 totalAssetsValue) {
        require(round != 0 && round <= navOracle.latestRound(), "CV:INR");
        totalAssetsValue = _recomputedTotalAssetsFromOraclePrices();
        latestTotalAssetsValue = totalAssetsValue;
        totalAssetsByRound[round] = totalAssetsValue;
        emit TotalAssetsUpdated(round, totalAssetsValue);
    }

    function _totalAssetsForRound(uint256 round) internal view returns (uint256 totalAssetsValue) {
        totalAssetsValue = totalAssetsByRound[round];
        require(totalAssetsValue > 0 || totalSupply() == 0, "CV:ZN");
    }

    /// User's share = Invested Amount / Net Value per Share
    // = Invested Amount / (totalAssetsValue / totalSupply)
    // = Invested Amount * totalSupply / totalAssetsValue
    function _previewDeposit(uint256 assets18, uint256 totalAssetsValue_, uint256 supply) internal pure returns (uint256) {
        if (supply == 0 || totalAssetsValue_ == 0) return assets18;
        return assets18 * supply / totalAssetsValue_;
    }

}


// ===== FILE: lib/openzeppelin-contracts/contracts/token/ERC20/IERC20.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 value) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}


// ===== FILE: lib/openzeppelin-contracts/contracts/token/ERC20/extensions/IERC20Metadata.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "../IERC20.sol";

interface IERC20Metadata is IERC20 {
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
}


// ===== FILE: lib/openzeppelin-contracts/contracts/token/ERC20/utils/SafeERC20.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "../IERC20.sol";

library SafeERC20 {
    function safeTransfer(IERC20 token, address to, uint256 value) internal {
        _call(token, abi.encodeWithSelector(token.transfer.selector, to, value));
    }

    function safeTransferFrom(IERC20 token, address from, address to, uint256 value) internal {
        _call(token, abi.encodeWithSelector(token.transferFrom.selector, from, to, value));
    }

    function safeApprove(IERC20 token, address spender, uint256 value) internal {
        _call(token, abi.encodeWithSelector(token.approve.selector, spender, value));
    }

    function forceApprove(IERC20 token, address spender, uint256 value) internal {
        safeApprove(token, spender, 0);
        safeApprove(token, spender, value);
    }

    function _call(IERC20 token, bytes memory data) private {
        (bool ok, bytes memory ret) = address(token).call(data);
        require(ok && (ret.length == 0 || abi.decode(ret, (bool))), "SafeERC20: failed");
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/Pausable.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

abstract contract Pausable {
    bool private _paused;

    event Paused(address account);
    event Unpaused(address account);

    modifier whenNotPaused() {
        require(!_paused, "Pausable: paused");
        _;
    }

    modifier whenPaused() {
        require(_paused, "Pausable: not paused");
        _;
    }

    function paused() public view virtual returns (bool) {
        return _paused;
    }

    function _pause() internal virtual whenNotPaused {
        _paused = true;
        emit Paused(msg.sender);
    }

    function _unpause() internal virtual whenPaused {
        _paused = false;
        emit Unpaused(msg.sender);
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/ReentrancyGuard.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

abstract contract ReentrancyGuard {
    uint256 private constant NOT_ENTERED = 1;
    uint256 private constant ENTERED = 2;
    uint256 private _status = NOT_ENTERED;

    modifier nonReentrant() {
        require(_status != ENTERED, "ReentrancyGuard: reentrant call");
        _status = ENTERED;
        _;
        _status = NOT_ENTERED;
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/introspection/IERC165.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC165 {
    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}


// ===== FILE: src/base/CoreAccess.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { IVaultAuthority } from "../interfaces/IVaultAuthority.sol";
import { Roles } from "../libraries/Roles.sol";
import { UUPSUpgradeable } from "oz-v5/proxy/utils/UUPSUpgradeable.sol";

/// @title CoreAccess
abstract contract CoreAccess is UUPSUpgradeable {
    IVaultAuthority public authority;
    bool private _coreAccessInitialized;

    event AuthoritySet(address indexed authority);

    constructor(address authority_) {
        if (authority_ == address(0)) {
            _coreAccessInitialized = true;
        } else {
            _initializeCoreAccess(authority_);
        }
    }

    function _initializeCoreAccess(address authority_) internal {
        require(!_coreAccessInitialized, "CoreAccess: initialized");
        require(authority_ != address(0), "CoreAccess: zero authority");
        _coreAccessInitialized = true;
        authority = IVaultAuthority(authority_);
        emit AuthoritySet(authority_);
    }

    modifier onlyAdmin() {
        _checkRole(Roles.DEFAULT_ADMIN_ROLE, msg.sender);
        _;
    }

    modifier onlySuperAdmin() {
        _checkRole(Roles.SUPER_ADMIN_ROLE, msg.sender);
        _;
    }

    modifier onlyRole(bytes32 role) {
        _checkRole(role, msg.sender);
        _;
    }

    function _authorizeUpgrade(address) internal override onlySuperAdmin { }

    function hasRole(bytes32 role, address account) public view returns (bool) {
        return address(authority) != address(0) && authority.hasRole(role, account);
    }

    function _checkRole(bytes32 role, address account) internal view {
        require(hasRole(role, account), "CoreAccess: missing role");
    }
}


// ===== FILE: src/interfaces/IAssetCustodian.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { BatchHint, RebalanceBuy, RebalanceTrigger, Trade } from "./Types.sol";

interface IAssetCustodian {
    function previewNextGroupId() external view returns (bytes32);

    function executeDeposit(uint256 reqId, uint256 totalUsdc, BatchHint calldata hint)
        external
        returns (bytes32 groupId);

    function executeRedeem(
        uint256 reqId,
        address[] calldata etfTokens,
        uint256[] calldata amounts,
        BatchHint calldata hint
    ) external returns (bytes32 groupId);

    function executeRebalance(
        bytes32 rbId,
        Trade[] calldata sells,
        RebalanceBuy[] calldata buys,
        RebalanceTrigger trigger,
        BatchHint calldata hint
    ) external returns (bytes32 groupId);

}


// ===== FILE: src/interfaces/IBlocklist.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IBlocklist {
    function isBlocked(address account) external view returns (bool);
}


// ===== FILE: src/interfaces/IFeeManager.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IFeeManager {
    function startAccrual() external;
    function accrueManagementFee(uint256 totalAssetsValue, uint256 effectiveSupply) external returns (uint256 shares);
    function claimManagementFee() external returns (uint256 shares);
    function claimPerformanceFee() external returns (uint256 shares);
    function claimAllFees() external returns (uint256 shares);
    function previewPerformanceFee(uint256 totalAssetsValue, uint256 effectiveSupply)
        external
        view
        returns (uint256 currentSharePrice, uint256 threshold, uint256 feeShares);
    function settlePerformanceFee(uint256 totalAssetsValue, uint256 effectiveSupply)
        external
        returns (uint256 shares, uint256 newHighWaterMark);
    function uncollectedFeeInShares() external view returns (uint256);
    function unclaimedManagementFeeShares() external view returns (uint256);
    function claimedManagementFeeShares() external view returns (uint256);
    function unclaimedPerformanceFeeShares() external view returns (uint256);
    function claimedPerformanceFeeShares() external view returns (uint256);
    function effectiveTotalSupply(uint256 totalSupply) external view returns (uint256);
}


// ===== FILE: src/interfaces/IERC7575.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC165} from "@openzeppelin/contracts/utils/introspection/IERC165.sol";
import { SignedPriceBundle } from "./Types.sol";


interface IERC7575 is IERC165 {
    /// @notice 返回该 Vault 对应的外部 ShareToken。
    function share() external view returns (address shareTokenAddress);
    function asset() external view returns (address assetTokenAddress);
    function totalAssets() external view returns (uint256 totalManagedAssets);
    function convertToShares(uint256 assets, SignedPriceBundle calldata priceBundle) external view returns (uint256 shares);
    function convertToAssets(uint256 shares, SignedPriceBundle calldata priceBundle) external view returns (uint256 assets);
}


// ===== FILE: src/interfaces/INAVOracle.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {SignedPriceBundle} from "./Types.sol";

interface INAVOracle {
    function ingest(SignedPriceBundle calldata bundle) external returns (uint64 round);
    function latestRound() external view returns (uint64);
    function priceOf(address asset) external view returns (uint256);
    function domainSeparator() external view returns (bytes32);
    function hashBundle(SignedPriceBundle calldata bundle) external view returns (bytes32);
    function recoverBundleSigner(SignedPriceBundle calldata bundle)
        external
        view
        returns (bytes32 digest, address recovered, bool signerMatches, bool recoveredTrusted);
    function recomputeTotalAssetsFromBundle(
        address vault,
        address usdc,
        SignedPriceBundle calldata bundle
    ) external view returns (uint256 totalAssetsValue);
}


// ===== FILE: src/interfaces/IRebalanceManager.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { RebalanceBuy, RebalanceTrigger, OndoTarget } from "./Types.sol";

interface IRebalanceManager {
    function setCoreVault(address coreVault) external;
    function setPassiveRebalanceInterval(uint256 interval) external;
    function prepareActiveRebalance(
        uint256 sellCount,
        uint256 buyCount,
        RebalanceBuy[] calldata buys,
        address[] calldata targetTokens,
        uint256[] calldata targetWeights
    ) external returns (bytes32 rbId);
    function preparePassiveRebalance(
        uint256 sellCount,
        uint256 buyCount,
        RebalanceTrigger trigger,
        uint8 riskLevel,
        bytes32 reasonHash,
        RebalanceBuy[] calldata buys,
        address[] calldata targetTokens,
        uint256[] calldata targetWeights
    ) external returns (bytes32 rbId);
    function validateRebalanceFilled(bytes32 rbId, address[] calldata tokens, uint256[] calldata amounts) external view;
    function settleRebalance(bytes32 rbId) external;
    function settleRebalanceFailure(bytes32 rbId) external;

    // Settlement entry points (called by AssetCustodian via CoreVault)
    function settleRebalanceComplete(
        bytes32 rbId,
        address[] calldata tokens,
        uint256[] calldata amounts
    ) external;

    function settleRebalanceFailed(bytes32 rbId) external;

    function isSettled(bytes32 rbId) external view returns (bool);

    // Pending Ondo Targets Management
    function savePendingOndoTargets(
        bytes32 rbId,
        address[] calldata targetTokens,
        uint256[] calldata targetWeights
    ) external;

    function getPendingOndoTargets(bytes32 rbId) external view returns (OndoTarget[] memory);

    function clearPendingOndoTargets(bytes32 rbId) external;

    function activeRebalanceId() external view returns (bytes32);
    function rebalanceTrigger(bytes32 rbId) external view returns (RebalanceTrigger);
    function rebalanceRiskLevel(bytes32 rbId) external view returns (uint8);
    function rebalanceReasonHash(bytes32 rbId) external view returns (bytes32);
    function minPassiveRebalanceInterval() external view returns (uint256);
    function lastPassiveRebalanceAt() external view returns (uint256);
}


// ===== FILE: src/interfaces/IRequestManager.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { RequestStatus } from "./Types.sol";

interface IRequestManager {
    struct RequestData {
        address owner;
        bool isDeposit;
        uint256 assets;
        uint256 shares;
        uint256 settledShares;
        uint256 refundAssets;
        uint256 navRound;
        uint256 totalAssetsSnapshot;
        uint256 supplySnapshot;
        RequestStatus status;
        bytes32 groupId;
    }

    struct DepositSettlement {
        address owner;
        uint256 spent;
        uint256 shares;
        uint256 refund;
    }

    struct RedeemSettlement {
        address owner;
        uint256 requestedShares;
        uint256 fulfilledShares;
        uint256 returnedShares;
        uint256 payout;
    }

    struct FailureSettlement {
        address owner;
        bool isDeposit;
        uint256 shares;
        uint256 refund;
    }

    function setCoreVault(address coreVault) external;
    function createDepositRequest(
        address owner,
        uint256 assets,
        uint256 shares,
        uint256 navRound,
        uint256 totalAssetsSnapshot,
        uint256 supplySnapshot
    ) external returns (uint256 reqId);
    function createRedeemRequest(
        address owner,
        uint256 shares,
        uint256 navRound,
        uint256 totalAssetsSnapshot,
        uint256 supplySnapshot,
        uint256 expectedAssets
    ) external returns (uint256 reqId);
    function setGroupId(uint256 reqId, bytes32 groupId) external;
    function validateDepositRequest(bytes32 groupId, uint256 reqId) external view;
    function settleDepositRequest(uint256 reqId, uint256 refundUsdc)
        external
        returns (DepositSettlement memory result);
    function validateRedeemRequest(bytes32 groupId, uint256 reqId) external view returns (uint256 totalShares);
    function settleRedeemRequest(
        uint256 reqId,
        uint256 usdcReceived
    ) external returns (RedeemSettlement memory result);
    function settleFailure(bytes32 groupId, uint256 reqId, uint256 returnedUsdc)
        external
        returns (FailureSettlement memory result);

    // Settlement entry points (called by AssetCustodian via CoreVault)
    function settleDeposit(
        bytes32 groupId,
        uint256 reqId,
        uint256 refundUsdc
    ) external returns (DepositSettlement memory result);

    function settleRedeem(
        bytes32 groupId,
        uint256 reqId,
        uint256 usdcReceived
    ) external returns (RedeemSettlement memory result);

    function settleRequestFailure(
        bytes32 groupId,
        uint256 reqId,
        uint256 returnedUsdc
    ) external returns (FailureSettlement memory result);

    function isSettled(bytes32 groupId) external view returns (bool);

    function getRequest(uint256 reqId) external view returns (RequestData memory);
    function userRequestCount(address owner) external view returns (uint256);
    function userRequestIdAt(address owner, uint256 index) external view returns (uint256);
    function userEscrowedShares(address owner) external view returns (uint256);
    function escrowedShares() external view returns (uint256);
}


// ===== FILE: src/interfaces/IShareToken.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC165} from "@openzeppelin/contracts/utils/introspection/IERC165.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";

interface IShareToken is IERC20, IERC165 {
    event VaultAdded(address indexed asset, address indexed vault);
    event VaultRemoved(address indexed asset, address indexed vault);

    function mint(address to, uint256 amount) external;
    function burn(address from, uint256 amount) external;
    function addVault(address asset, address vault_) external;
    function removeVault(address asset) external;
    function vault(address asset) external view returns (address);
    function asset(address vault_) external view returns (address);
    function isVault(address vault_) external view returns (bool);
}


// ===== FILE: src/interfaces/IWhitelist.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IWhitelist {
    function isWhitelisted(address account) external view returns (bool);
    function isAssetAllowed(address asset) external view returns (bool);
}


// ===== FILE: src/interfaces/IVaultSettlement.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IVaultSettlement {
    function settleDeposit(
        bytes32 groupId,
        uint256 reqId,
        address[] calldata etfTokens,
        uint256[] calldata amounts,
        uint256 refundUsdc,
        uint256 navRound
    ) external;

    function settleRedeem(bytes32 groupId, uint256 reqId, uint256 usdcReceived, uint256 navRound)
        external;
    function settleRebalance(bytes32 rbId, address[] calldata tokens, uint256[] calldata amounts)
        external;
    function settleRebalanceFailure(bytes32 rbId, bytes calldata reason) external;
    function settleFailure(bytes32 groupId, uint256 reqId, uint256 returnedUsdc, bytes calldata reason)
        external;
}


// ===== FILE: src/interfaces/Types.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

struct PriceEntry {
    address asset;
    uint256 price;
    uint8 priceDecimals;
    uint64 asOfTimestamp;
}

struct SignedPriceBundle {
    uint64 round;
    uint64 deadline;
    PriceEntry[] entries;
    bytes32 assetSetDigest;
    address signer;
    uint8 v;
    bytes32 r;
    bytes32 s;
}

struct OndoTarget {
    address token;
    uint16 bps;
}

struct BatchHint {
    uint16 maxSlippageBps;
    uint16 maxLossPercentBps;
}

struct Trade {
    address tokenIn;
    address tokenOut;
    uint256 amountIn;
    uint256 minAmountOut;
}


struct RebalanceBuy {
    address tokenIn;
    address tokenOut;
    uint256 amountIn;
    uint256 minAmountOut;
}

enum RequestStatus {
    Executing,
    Completed,
    Failed
}

enum BatchStatus {
    PendingProxy,
    Settled,
    Failed
}

enum RebalanceTrigger {
    Active,
    PassivePeriodic,
    PassiveRiskEvent
}

struct PassiveRebalanceParams {
    Trade[] sells;
    RebalanceBuy[] buys;
    address[] targetTokens;
    uint256[] targetWeights;
    RebalanceTrigger trigger;
    uint8 riskLevel;
    bytes32 reasonHash;
    SignedPriceBundle priceBundle;
}


// ===== FILE: src/libraries/FixedPoint.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { Math } from "@openzeppelin/contracts/utils/math/Math.sol";

/// @title FixedPoint
library FixedPoint {
    uint256 internal constant WAD = 1e18;
    uint256 internal constant BPS = 10_000;
    uint256 internal constant USDC_SCALE = 1e12;

    function usdcToWad(uint256 amount) internal pure returns (uint256) {
        return amount * USDC_SCALE;
    }

    function wadToUsdc(uint256 amount) internal pure returns (uint256) {
        return amount / USDC_SCALE;
    }

    function mulDiv(uint256 x, uint256 y, uint256 denominator) internal pure returns (uint256) {
        return Math.mulDiv(x, y, denominator);
    }

    function bpsOf(uint256 amount, uint256 bps) internal pure returns (uint256) {
        return amount * bps / BPS;
    }
}


// ===== FILE: src/libraries/Roles.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title Roles
library Roles {
    bytes32 internal constant DEFAULT_ADMIN_ROLE = 0x00;
    // 0x7613a25ecc738585a232ad50a301178f12b3ba8887d13e138b523c4269c47689
    bytes32 internal constant SUPER_ADMIN_ROLE = keccak256("SUPER_ADMIN_ROLE");
    // 0x139c2898040ef16910dc9f44dc697df79363da767d8bc92f2e310312b816e46d
    bytes32 internal constant PAUSE_ROLE = keccak256("PAUSE_ROLE");
    // 0x0aec9b08bc8c2cb62a91f52d33e3d77da4b3f3a63fc8b542a93abe3902ba929c
    bytes32 internal constant CURATOR = keccak256("CURATOR");
    // 0x390ae1081063aad083f9b4572bbfcc286af7814a39b668db2927a1186deec81b
    bytes32 internal constant MONITOR = keccak256("MONITOR");
    // 0x523a704056dcd17bcf83bed8b68c59416dac1119be77755efe3bde0a64e46e0c
    bytes32 internal constant OPERATOR = keccak256("OPERATOR");
    // 0x8a639721758084b2ff869ed05cee658aab46e6bdc69bca1b006bae9f55613c59
    bytes32 internal constant FEE_CONTROLLER = keccak256("FEE_CONTROLLER");
    // 0x262c70cb68844873654dc54487b634cb00850c1e13c785cd0d96a2b89b829472
    bytes32 internal constant TOKEN_MINTER = keccak256("TOKEN_MINTER");
    // 0x47bc4cfe7b2bde7da37514f11daac8409232278f94544d2e3175143878762e9f
    bytes32 internal constant NAV_CALLER = keccak256("NAV_CALLER");
    //
    bytes32 internal constant ONDO_TARGET_MANAGER = keccak256("ONDO_TARGET_MANAGER");
}


// ===== FILE: src/interfaces/IVaultAuthority.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IVaultAuthority {
    event RoleGranted(bytes32 indexed role, address indexed account, address indexed sender);
    event RoleRevoked(bytes32 indexed role, address indexed account, address indexed sender);

    function hasRole(bytes32 role, address account) external view returns (bool);
    function grantAdmin(address account) external;
    function revokeAdmin(address account) external;
    function grantRole(bytes32 role, address account) external;
    function revokeRole(bytes32 role, address account) external;
    function renounceRole(bytes32 role, address account) external;
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/proxy/utils/UUPSUpgradeable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.5.0) (proxy/utils/UUPSUpgradeable.sol)

pragma solidity ^0.8.22;

import {IERC1822Proxiable} from "../../interfaces/draft-IERC1822.sol";
import {ERC1967Utils} from "../ERC1967/ERC1967Utils.sol";

/**
 * @dev An upgradeability mechanism designed for UUPS proxies. The functions included here can perform an upgrade of an
 * {ERC1967Proxy}, when this contract is set as the implementation behind such a proxy.
 *
 * A security mechanism ensures that an upgrade does not turn off upgradeability accidentally, although this risk is
 * reinstated if the upgrade retains upgradeability but removes the security mechanism, e.g. by replacing
 * `UUPSUpgradeable` with a custom implementation of upgrades.
 *
 * The {_authorizeUpgrade} function must be overridden to include access restriction to the upgrade mechanism.
 *
 * @custom:stateless
 */
abstract contract UUPSUpgradeable is IERC1822Proxiable {
    /// @custom:oz-upgrades-unsafe-allow state-variable-immutable
    address private immutable __self = address(this);

    /**
     * @dev The version of the upgrade interface of the contract. If this getter is missing, both `upgradeTo(address)`
     * and `upgradeToAndCall(address,bytes)` are present, and `upgradeTo` must be used if no function should be called,
     * while `upgradeToAndCall` will invoke the `receive` function if the second argument is the empty byte string.
     * If the getter returns `"5.0.0"`, only `upgradeToAndCall(address,bytes)` is present, and the second argument must
     * be the empty byte string if no function should be called, making it impossible to invoke the `receive` function
     * during an upgrade.
     */
    string public constant UPGRADE_INTERFACE_VERSION = "5.0.0";

    /**
     * @dev The call is from an unauthorized context.
     */
    error UUPSUnauthorizedCallContext();

    /**
     * @dev The storage `slot` is unsupported as a UUID.
     */
    error UUPSUnsupportedProxiableUUID(bytes32 slot);

    /**
     * @dev Check that the execution is being performed through a delegatecall call and that the execution context is
     * a proxy contract with an implementation (as defined in ERC-1967) pointing to self. This should only be the case
     * for UUPS and transparent proxies that are using the current contract as their implementation. Execution of a
     * function through ERC-1167 minimal proxies (clones) would not normally pass this test, but is not guaranteed to
     * fail.
     */
    modifier onlyProxy() {
        _checkProxy();
        _;
    }

    /**
     * @dev Check that the execution is not being performed through a delegate call. This allows a function to be
     * callable on the implementing contract but not through proxies.
     */
    modifier notDelegated() {
        _checkNotDelegated();
        _;
    }

    /**
     * @dev Implementation of the ERC-1822 {proxiableUUID} function. This returns the storage slot used by the
     * implementation. It is used to validate the implementation's compatibility when performing an upgrade.
     *
     * IMPORTANT: A proxy pointing at a proxiable contract should not be considered proxiable itself, because this risks
     * bricking a proxy that upgrades to it, by delegating to itself until out of gas. Thus it is critical that this
     * function revert if invoked through a proxy. This is guaranteed by the `notDelegated` modifier.
     */
    function proxiableUUID() external view notDelegated returns (bytes32) {
        return ERC1967Utils.IMPLEMENTATION_SLOT;
    }

    /**
     * @dev Upgrade the implementation of the proxy to `newImplementation`, and subsequently execute the function call
     * encoded in `data`.
     *
     * Calls {_authorizeUpgrade}.
     *
     * Emits an {Upgraded} event.
     *
     * @custom:oz-upgrades-unsafe-allow-reachable delegatecall
     */
    function upgradeToAndCall(address newImplementation, bytes memory data) public payable virtual onlyProxy {
        _authorizeUpgrade(newImplementation);
        _upgradeToAndCallUUPS(newImplementation, data);
    }

    /**
     * @dev Reverts if the execution is not performed via delegatecall or the execution
     * context is not of a proxy with an ERC-1967 compliant implementation pointing to self.
     */
    function _checkProxy() internal view virtual {
        if (
            address(this) == __self || // Must be called through delegatecall
            ERC1967Utils.getImplementation() != __self // Must be called through an active proxy
        ) {
            revert UUPSUnauthorizedCallContext();
        }
    }

    /**
     * @dev Reverts if the execution is performed via delegatecall.
     * See {notDelegated}.
     */
    function _checkNotDelegated() internal view virtual {
        if (address(this) != __self) {
            // Must not be called through delegatecall
            revert UUPSUnauthorizedCallContext();
        }
    }

    /**
     * @dev Function that should revert when `msg.sender` is not authorized to upgrade the contract. Called by
     * {upgradeToAndCall}.
     *
     * Normally, this function will use an xref:access.adoc[access control] modifier such as {Ownable-onlyOwner}.
     *
     * ```solidity
     * function _authorizeUpgrade(address) internal onlyOwner {}
     * ```
     */
    function _authorizeUpgrade(address newImplementation) internal virtual;

    /**
     * @dev Performs an implementation upgrade with a security check for UUPS proxies, and additional setup call.
     *
     * As a security check, {proxiableUUID} is invoked in the new implementation, and the return value
     * is expected to be the implementation slot in ERC-1967.
     *
     * Emits an {IERC1967-Upgraded} event.
     */
    function _upgradeToAndCallUUPS(address newImplementation, bytes memory data) private {
        try IERC1822Proxiable(newImplementation).proxiableUUID() returns (bytes32 slot) {
            if (slot != ERC1967Utils.IMPLEMENTATION_SLOT) {
                revert UUPSUnsupportedProxiableUUID(slot);
            }
            ERC1967Utils.upgradeToAndCall(newImplementation, data);
        } catch {
            // The implementation is not UUPS
            revert ERC1967Utils.ERC1967InvalidImplementation(newImplementation);
        }
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/math/Math.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title Math
/// @notice OpenZeppelin Math 最小兼容库，本项目当前只需要全精度 mulDiv。
library Math {
    /// @notice 计算 floor(x * y / denominator)，支持 512 位中间乘积，避免 x * y 溢出。
    function mulDiv(uint256 x, uint256 y, uint256 denominator) internal pure returns (uint256 result) {
        unchecked {
            uint256 prod0;
            uint256 prod1;
            assembly {
                let mm := mulmod(x, y, not(0))
                prod0 := mul(x, y)
                prod1 := sub(sub(mm, prod0), lt(mm, prod0))
            }

            if (prod1 == 0) {
                require(denominator > 0, "Math: div0");
                return prod0 / denominator;
            }

            require(denominator > prod1, "Math: overflow");

            uint256 remainder;
            assembly {
                remainder := mulmod(x, y, denominator)
                prod1 := sub(prod1, gt(remainder, prod0))
                prod0 := sub(prod0, remainder)
            }

            uint256 twos = denominator & (0 - denominator);
            assembly {
                denominator := div(denominator, twos)
                prod0 := div(prod0, twos)
                twos := add(div(sub(0, twos), twos), 1)
            }
            prod0 |= prod1 * twos;

            uint256 inverse = (3 * denominator) ^ 2;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;

            result = prod0 * inverse;
            return result;
        }
    }
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/interfaces/draft-IERC1822.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (interfaces/draft-IERC1822.sol)

pragma solidity >=0.4.16;

/**
 * @dev ERC-1822: Universal Upgradeable Proxy Standard (UUPS) documents a method for upgradeability through a simplified
 * proxy whose upgrades are fully controlled by the current implementation.
 */
interface IERC1822Proxiable {
    /**
     * @dev Returns the storage slot that the proxiable contract assumes is being used to store the implementation
     * address.
     *
     * IMPORTANT: A proxy pointing at a proxiable contract should not be considered proxiable itself, because this risks
     * bricking a proxy that upgrades to it, by delegating to itself until out of gas. Thus it is critical that this
     * function revert if invoked through a proxy.
     */
    function proxiableUUID() external view returns (bytes32);
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/proxy/ERC1967/ERC1967Utils.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.6.0) (proxy/ERC1967/ERC1967Utils.sol)

pragma solidity ^0.8.21;

import {IBeacon} from "../beacon/IBeacon.sol";
import {IERC1967} from "../../interfaces/IERC1967.sol";
import {Address} from "../../utils/Address.sol";
import {StorageSlot} from "../../utils/StorageSlot.sol";

/**
 * @dev This library provides getters and event emitting update functions for
 * https://eips.ethereum.org/EIPS/eip-1967[ERC-1967] slots.
 */
library ERC1967Utils {
    /**
     * @dev Storage slot with the address of the current implementation.
     * This is the keccak-256 hash of "eip1967.proxy.implementation" subtracted by 1.
     */
    // solhint-disable-next-line private-vars-leading-underscore
    bytes32 internal constant IMPLEMENTATION_SLOT = 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc;

    /**
     * @dev The `implementation` of the proxy is invalid.
     */
    error ERC1967InvalidImplementation(address implementation);

    /**
     * @dev The `admin` of the proxy is invalid.
     */
    error ERC1967InvalidAdmin(address admin);

    /**
     * @dev The `beacon` of the proxy is invalid.
     */
    error ERC1967InvalidBeacon(address beacon);

    /**
     * @dev An upgrade function sees `msg.value > 0` that may be lost.
     */
    error ERC1967NonPayable();

    /**
     * @dev Returns the current implementation address.
     */
    function getImplementation() internal view returns (address) {
        return StorageSlot.getAddressSlot(IMPLEMENTATION_SLOT).value;
    }

    /**
     * @dev Stores a new address in the ERC-1967 implementation slot.
     */
    function _setImplementation(address newImplementation) private {
        if (newImplementation.code.length == 0) {
            revert ERC1967InvalidImplementation(newImplementation);
        }
        StorageSlot.getAddressSlot(IMPLEMENTATION_SLOT).value = newImplementation;
    }

    /**
     * @dev Performs implementation upgrade with additional setup call if data is nonempty.
     * This function is payable only if the setup call is performed, otherwise `msg.value` is rejected
     * to avoid stuck value in the contract.
     *
     * Emits an {IERC1967-Upgraded} event.
     */
    function upgradeToAndCall(address newImplementation, bytes memory data) internal {
        _setImplementation(newImplementation);
        emit IERC1967.Upgraded(newImplementation);

        if (data.length > 0) {
            Address.functionDelegateCall(newImplementation, data);
        } else {
            _checkNonPayable();
        }
    }

    /**
     * @dev Storage slot with the admin of the contract.
     * This is the keccak-256 hash of "eip1967.proxy.admin" subtracted by 1.
     */
    // solhint-disable-next-line private-vars-leading-underscore
    bytes32 internal constant ADMIN_SLOT = 0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103;

    /**
     * @dev Returns the current admin.
     *
     * TIP: To get this value clients can read directly from the storage slot shown below (specified by ERC-1967) using
     * the https://ethereum.org/developers/docs/apis/json-rpc/#eth_getstorageat[`eth_getStorageAt`] RPC call.
     * `0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103`
     */
    function getAdmin() internal view returns (address) {
        return StorageSlot.getAddressSlot(ADMIN_SLOT).value;
    }

    /**
     * @dev Stores a new address in the ERC-1967 admin slot.
     */
    function _setAdmin(address newAdmin) private {
        if (newAdmin == address(0)) {
            revert ERC1967InvalidAdmin(address(0));
        }
        StorageSlot.getAddressSlot(ADMIN_SLOT).value = newAdmin;
    }

    /**
     * @dev Changes the admin of the proxy.
     *
     * Emits an {IERC1967-AdminChanged} event.
     */
    function changeAdmin(address newAdmin) internal {
        emit IERC1967.AdminChanged(getAdmin(), newAdmin);
        _setAdmin(newAdmin);
    }

    /**
     * @dev The storage slot of the UpgradeableBeacon contract which defines the implementation for this proxy.
     * This is the keccak-256 hash of "eip1967.proxy.beacon" subtracted by 1.
     */
    // solhint-disable-next-line private-vars-leading-underscore
    bytes32 internal constant BEACON_SLOT = 0xa3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50;

    /**
     * @dev Returns the current beacon.
     */
    function getBeacon() internal view returns (address) {
        return StorageSlot.getAddressSlot(BEACON_SLOT).value;
    }

    /**
     * @dev Stores a new beacon in the ERC-1967 beacon slot.
     */
    function _setBeacon(address newBeacon) private {
        if (newBeacon.code.length == 0) {
            revert ERC1967InvalidBeacon(newBeacon);
        }

        StorageSlot.getAddressSlot(BEACON_SLOT).value = newBeacon;

        address beaconImplementation = IBeacon(newBeacon).implementation();
        if (beaconImplementation.code.length == 0) {
            revert ERC1967InvalidImplementation(beaconImplementation);
        }
    }

    /**
     * @dev Change the beacon and trigger a setup call if data is nonempty.
     * This function is payable only if the setup call is performed, otherwise `msg.value` is rejected
     * to avoid stuck value in the contract.
     *
     * Emits an {IERC1967-BeaconUpgraded} event.
     *
     * CAUTION: Invoking this function has no effect on an instance of {BeaconProxy} since v5, since
     * it uses an immutable beacon without looking at the value of the ERC-1967 beacon slot for
     * efficiency.
     */
    function upgradeBeaconToAndCall(address newBeacon, bytes memory data) internal {
        _setBeacon(newBeacon);
        emit IERC1967.BeaconUpgraded(newBeacon);

        if (data.length > 0) {
            Address.functionDelegateCall(IBeacon(newBeacon).implementation(), data);
        } else {
            _checkNonPayable();
        }
    }

    /**
     * @dev Reverts if `msg.value` is not zero. It can be used to avoid `msg.value` stuck in the contract
     * if an upgrade doesn't perform an initialization call.
     */
    function _checkNonPayable() private {
        if (msg.value > 0) {
            revert ERC1967NonPayable();
        }
    }
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/proxy/beacon/IBeacon.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (proxy/beacon/IBeacon.sol)

pragma solidity >=0.4.16;

/**
 * @dev This is the interface that {BeaconProxy} expects of its beacon.
 */
interface IBeacon {
    /**
     * @dev Must return an address that can be used as a delegate call target.
     *
     * {UpgradeableBeacon} will check that this address is a contract.
     */
    function implementation() external view returns (address);
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/interfaces/IERC1967.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (interfaces/IERC1967.sol)

pragma solidity >=0.4.11;

/**
 * @dev ERC-1967: Proxy Storage Slots. This interface contains the events defined in the ERC.
 */
interface IERC1967 {
    /**
     * @dev Emitted when the implementation is upgraded.
     */
    event Upgraded(address indexed implementation);

    /**
     * @dev Emitted when the admin account has changed.
     */
    event AdminChanged(address previousAdmin, address newAdmin);

    /**
     * @dev Emitted when the beacon is changed.
     */
    event BeaconUpgraded(address indexed beacon);
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/utils/Address.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.5.0) (utils/Address.sol)

pragma solidity ^0.8.20;

import {Errors} from "./Errors.sol";
import {LowLevelCall} from "./LowLevelCall.sol";

/**
 * @dev Collection of functions related to the address type
 */
library Address {
    /**
     * @dev There's no code at `target` (it is not a contract).
     */
    error AddressEmptyCode(address target);

    /**
     * @dev Replacement for Solidity's `transfer`: sends `amount` wei to
     * `recipient`, forwarding all available gas and reverting on errors.
     *
     * https://eips.ethereum.org/EIPS/eip-1884[EIP1884] increases the gas cost
     * of certain opcodes, possibly making contracts go over the 2300 gas limit
     * imposed by `transfer`, making them unable to receive funds via
     * `transfer`. {sendValue} removes this limitation.
     *
     * https://consensys.net/diligence/blog/2019/09/stop-using-soliditys-transfer-now/[Learn more].
     *
     * IMPORTANT: because control is transferred to `recipient`, care must be
     * taken to not create reentrancy vulnerabilities. Consider using
     * {ReentrancyGuard} or the
     * https://solidity.readthedocs.io/en/v0.8.20/security-considerations.html#use-the-checks-effects-interactions-pattern[checks-effects-interactions pattern].
     */
    function sendValue(address payable recipient, uint256 amount) internal {
        if (address(this).balance < amount) {
            revert Errors.InsufficientBalance(address(this).balance, amount);
        }
        if (LowLevelCall.callNoReturn(recipient, amount, "")) {
            // call successful, nothing to do
            return;
        } else if (LowLevelCall.returnDataSize() > 0) {
            LowLevelCall.bubbleRevert();
        } else {
            revert Errors.FailedCall();
        }
    }

    /**
     * @dev Performs a Solidity function call using a low level `call`. A
     * plain `call` is an unsafe replacement for a function call: use this
     * function instead.
     *
     * If `target` reverts with a revert reason or custom error, it is bubbled
     * up by this function (like regular Solidity function calls). However, if
     * the call reverted with no returned reason, this function reverts with a
     * {Errors.FailedCall} error.
     *
     * Returns the raw returned data. To convert to the expected return value,
     * use https://solidity.readthedocs.io/en/latest/units-and-global-variables.html?highlight=abi.decode#abi-encoding-and-decoding-functions[`abi.decode`].
     *
     * Requirements:
     *
     * - `target` must be a contract.
     * - calling `target` with `data` must not revert.
     */
    function functionCall(address target, bytes memory data) internal returns (bytes memory) {
        return functionCallWithValue(target, data, 0);
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but also transferring `value` wei to `target`.
     *
     * Requirements:
     *
     * - the calling contract must have an ETH balance of at least `value`.
     * - the called Solidity function must be `payable`.
     */
    function functionCallWithValue(address target, bytes memory data, uint256 value) internal returns (bytes memory) {
        if (address(this).balance < value) {
            revert Errors.InsufficientBalance(address(this).balance, value);
        }
        bool success = LowLevelCall.callNoReturn(target, value, data);
        if (success && (LowLevelCall.returnDataSize() > 0 || target.code.length > 0)) {
            return LowLevelCall.returnData();
        } else if (success) {
            revert AddressEmptyCode(target);
        } else if (LowLevelCall.returnDataSize() > 0) {
            LowLevelCall.bubbleRevert();
        } else {
            revert Errors.FailedCall();
        }
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but performing a static call.
     */
    function functionStaticCall(address target, bytes memory data) internal view returns (bytes memory) {
        bool success = LowLevelCall.staticcallNoReturn(target, data);
        if (success && (LowLevelCall.returnDataSize() > 0 || target.code.length > 0)) {
            return LowLevelCall.returnData();
        } else if (success) {
            revert AddressEmptyCode(target);
        } else if (LowLevelCall.returnDataSize() > 0) {
            LowLevelCall.bubbleRevert();
        } else {
            revert Errors.FailedCall();
        }
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but performing a delegate call.
     */
    function functionDelegateCall(address target, bytes memory data) internal returns (bytes memory) {
        bool success = LowLevelCall.delegatecallNoReturn(target, data);
        if (success && (LowLevelCall.returnDataSize() > 0 || target.code.length > 0)) {
            return LowLevelCall.returnData();
        } else if (success) {
            revert AddressEmptyCode(target);
        } else if (LowLevelCall.returnDataSize() > 0) {
            LowLevelCall.bubbleRevert();
        } else {
            revert Errors.FailedCall();
        }
    }

    /**
     * @dev Tool to verify that a low level call to smart-contract was successful, and reverts if the target
     * was not a contract or bubbling up the revert reason (falling back to {Errors.FailedCall}) in case
     * of an unsuccessful call.
     *
     * NOTE: This function is DEPRECATED and may be removed in the next major release.
     */
    function verifyCallResultFromTarget(
        address target,
        bool success,
        bytes memory returndata
    ) internal view returns (bytes memory) {
        // only check if target is a contract if the call was successful and the return data is empty
        // otherwise we already know that it was a contract
        if (success && (returndata.length > 0 || target.code.length > 0)) {
            return returndata;
        } else if (success) {
            revert AddressEmptyCode(target);
        } else if (returndata.length > 0) {
            LowLevelCall.bubbleRevert(returndata);
        } else {
            revert Errors.FailedCall();
        }
    }

    /**
     * @dev Tool to verify that a low level call was successful, and reverts if it wasn't, either by bubbling the
     * revert reason or with a default {Errors.FailedCall} error.
     */
    function verifyCallResult(bool success, bytes memory returndata) internal pure returns (bytes memory) {
        if (success) {
            return returndata;
        } else if (returndata.length > 0) {
            LowLevelCall.bubbleRevert(returndata);
        } else {
            revert Errors.FailedCall();
        }
    }
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/utils/StorageSlot.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/StorageSlot.sol)
// This file was procedurally generated from scripts/generate/templates/StorageSlot.js.

pragma solidity ^0.8.20;

/**
 * @dev Library for reading and writing primitive types to specific storage slots.
 *
 * Storage slots are often used to avoid storage conflict when dealing with upgradeable contracts.
 * This library helps with reading and writing to such slots without the need for inline assembly.
 *
 * The functions in this library return Slot structs that contain a `value` member that can be used to read or write.
 *
 * Example usage to set ERC-1967 implementation slot:
 * ```solidity
 * contract ERC1967 {
 *     // Define the slot. Alternatively, use the SlotDerivation library to derive the slot.
 *     bytes32 internal constant _IMPLEMENTATION_SLOT = 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc;
 *
 *     function _getImplementation() internal view returns (address) {
 *         return StorageSlot.getAddressSlot(_IMPLEMENTATION_SLOT).value;
 *     }
 *
 *     function _setImplementation(address newImplementation) internal {
 *         require(newImplementation.code.length > 0);
 *         StorageSlot.getAddressSlot(_IMPLEMENTATION_SLOT).value = newImplementation;
 *     }
 * }
 * ```
 *
 * TIP: Consider using this library along with {SlotDerivation}.
 */
library StorageSlot {
    struct AddressSlot {
        address value;
    }

    struct BooleanSlot {
        bool value;
    }

    struct Bytes32Slot {
        bytes32 value;
    }

    struct Uint256Slot {
        uint256 value;
    }

    struct Int256Slot {
        int256 value;
    }

    struct StringSlot {
        string value;
    }

    struct BytesSlot {
        bytes value;
    }

    /**
     * @dev Returns an `AddressSlot` with member `value` located at `slot`.
     */
    function getAddressSlot(bytes32 slot) internal pure returns (AddressSlot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns a `BooleanSlot` with member `value` located at `slot`.
     */
    function getBooleanSlot(bytes32 slot) internal pure returns (BooleanSlot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns a `Bytes32Slot` with member `value` located at `slot`.
     */
    function getBytes32Slot(bytes32 slot) internal pure returns (Bytes32Slot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns a `Uint256Slot` with member `value` located at `slot`.
     */
    function getUint256Slot(bytes32 slot) internal pure returns (Uint256Slot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns a `Int256Slot` with member `value` located at `slot`.
     */
    function getInt256Slot(bytes32 slot) internal pure returns (Int256Slot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns a `StringSlot` with member `value` located at `slot`.
     */
    function getStringSlot(bytes32 slot) internal pure returns (StringSlot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns an `StringSlot` representation of the string storage pointer `store`.
     */
    function getStringSlot(string storage store) internal pure returns (StringSlot storage r) {
        assembly ("memory-safe") {
            r.slot := store.slot
        }
    }

    /**
     * @dev Returns a `BytesSlot` with member `value` located at `slot`.
     */
    function getBytesSlot(bytes32 slot) internal pure returns (BytesSlot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns an `BytesSlot` representation of the bytes storage pointer `store`.
     */
    function getBytesSlot(bytes storage store) internal pure returns (BytesSlot storage r) {
        assembly ("memory-safe") {
            r.slot := store.slot
        }
    }
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/utils/Errors.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/Errors.sol)

pragma solidity ^0.8.20;

/**
 * @dev Collection of common custom errors used in multiple contracts
 *
 * IMPORTANT: Backwards compatibility is not guaranteed in future versions of the library.
 * It is recommended to avoid relying on the error API for critical functionality.
 *
 * _Available since v5.1._
 */
library Errors {
    /**
     * @dev The ETH balance of the account is not enough to perform the operation.
     */
    error InsufficientBalance(uint256 balance, uint256 needed);

    /**
     * @dev A call to an address target failed. The target may have reverted.
     */
    error FailedCall();

    /**
     * @dev The deployment failed.
     */
    error FailedDeployment();

    /**
     * @dev A necessary precompile is missing.
     */
    error MissingPrecompile(address);
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/utils/LowLevelCall.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.6.0) (utils/LowLevelCall.sol)

pragma solidity ^0.8.20;

/**
 * @dev Library of low level call functions that implement different calling strategies to deal with the return data.
 *
 * WARNING: Using this library requires an advanced understanding of Solidity and how the EVM works. It is recommended
 * to use the {Address} library instead.
 */
library LowLevelCall {
    /// @dev Performs a Solidity function call using a low level `call` and ignoring the return data.
    function callNoReturn(address target, bytes memory data) internal returns (bool success) {
        return callNoReturn(target, 0, data);
    }

    /// @dev Same as {callNoReturn-address-bytes}, but allows specifying the value to be sent in the call.
    function callNoReturn(address target, uint256 value, bytes memory data) internal returns (bool success) {
        assembly ("memory-safe") {
            success := call(gas(), target, value, add(data, 0x20), mload(data), 0x00, 0x00)
        }
    }

    /// @dev Performs a Solidity function call using a low level `call` and returns the first 64 bytes of the result
    /// in the scratch space of memory. Useful for functions that return a tuple with two single-word values.
    ///
    /// WARNING: Do not assume that the results are zero if `success` is false. Memory can be already allocated
    /// and this function doesn't zero it out.
    function callReturn64Bytes(
        address target,
        bytes memory data
    ) internal returns (bool success, bytes32 result1, bytes32 result2) {
        return callReturn64Bytes(target, 0, data);
    }

    /// @dev Same as {callReturn64Bytes-address-bytes}, but allows specifying the value to be sent in the call.
    function callReturn64Bytes(
        address target,
        uint256 value,
        bytes memory data
    ) internal returns (bool success, bytes32 result1, bytes32 result2) {
        assembly ("memory-safe") {
            success := call(gas(), target, value, add(data, 0x20), mload(data), 0x00, 0x40)
            result1 := mload(0x00)
            result2 := mload(0x20)
        }
    }

    /// @dev Performs a Solidity function call using a low level `staticcall` and ignoring the return data.
    function staticcallNoReturn(address target, bytes memory data) internal view returns (bool success) {
        assembly ("memory-safe") {
            success := staticcall(gas(), target, add(data, 0x20), mload(data), 0x00, 0x00)
        }
    }

    /// @dev Performs a Solidity function call using a low level `staticcall` and returns the first 64 bytes of the result
    /// in the scratch space of memory. Useful for functions that return a tuple with two single-word values.
    ///
    /// WARNING: Do not assume that the results are zero if `success` is false. Memory can be already allocated
    /// and this function doesn't zero it out.
    function staticcallReturn64Bytes(
        address target,
        bytes memory data
    ) internal view returns (bool success, bytes32 result1, bytes32 result2) {
        assembly ("memory-safe") {
            success := staticcall(gas(), target, add(data, 0x20), mload(data), 0x00, 0x40)
            result1 := mload(0x00)
            result2 := mload(0x20)
        }
    }

    /// @dev Performs a Solidity function call using a low level `delegatecall` and ignoring the return data.
    function delegatecallNoReturn(address target, bytes memory data) internal returns (bool success) {
        assembly ("memory-safe") {
            success := delegatecall(gas(), target, add(data, 0x20), mload(data), 0x00, 0x00)
        }
    }

    /// @dev Performs a Solidity function call using a low level `delegatecall` and returns the first 64 bytes of the result
    /// in the scratch space of memory. Useful for functions that return a tuple with two single-word values.
    ///
    /// WARNING: Do not assume that the results are zero if `success` is false. Memory can be already allocated
    /// and this function doesn't zero it out.
    function delegatecallReturn64Bytes(
        address target,
        bytes memory data
    ) internal returns (bool success, bytes32 result1, bytes32 result2) {
        assembly ("memory-safe") {
            success := delegatecall(gas(), target, add(data, 0x20), mload(data), 0x00, 0x40)
            result1 := mload(0x00)
            result2 := mload(0x20)
        }
    }

    /// @dev Returns the size of the return data buffer.
    function returnDataSize() internal pure returns (uint256 size) {
        assembly ("memory-safe") {
            size := returndatasize()
        }
    }

    /// @dev Returns a buffer containing the return data from the last call.
    function returnData() internal pure returns (bytes memory result) {
        assembly ("memory-safe") {
            result := mload(0x40)
            mstore(result, returndatasize())
            returndatacopy(add(result, 0x20), 0x00, returndatasize())
            mstore(0x40, add(result, add(0x20, returndatasize())))
        }
    }

    /// @dev Revert with the return data from the last call.
    function bubbleRevert() internal pure {
        assembly ("memory-safe") {
            let fmp := mload(0x40)
            returndatacopy(fmp, 0x00, returndatasize())
            revert(fmp, returndatasize())
        }
    }

    function bubbleRevert(bytes memory returndata) internal pure {
        assembly ("memory-safe") {
            revert(add(returndata, 0x20), mload(returndata))
        }
    }
}
