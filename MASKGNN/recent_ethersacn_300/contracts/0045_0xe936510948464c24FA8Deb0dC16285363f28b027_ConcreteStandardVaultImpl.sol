// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/implementation/ConcreteStandardVaultImpl.sol =====
// SPDX-License-Identifier: AGPL-3.0
pragma solidity ^0.8.24;

/**
 * @title ConcreteStandardVaultImpl
 * @notice ERC-4626 upgradeable standard vault implementation for the Concrete Earn V2 protocol.
 *         Holds an underlying ERC20 asset and exposes deposit/mint/withdraw/redeem flows.
 *         Integrates with strategy modules to route assets to yield sources.
 *
 * @author Blueprint Finance
 * @custom:protocol Concrete Earn V2
 * @custom:oz-upgrades Use OZ Upgradeable patterns and eip7201 storage layout
 * @custom:source on request
 * @custom:audits on request
 * @custom:license AGPL-3.0
 */

// ─────────────────────────────────────────────────────────────────────────────
// External dependencies
// ─────────────────────────────────────────────────────────────────────────────
import {
    AccessControlEnumerableUpgradeable
} from "@openzeppelin-upgradeable/access/extensions/AccessControlEnumerableUpgradeable.sol";
import {
    ERC4626Upgradeable,
    IERC20,
    IERC4626,
    Math,
    SafeERC20
} from "@openzeppelin-upgradeable/token/ERC20/extensions/ERC4626Upgradeable.sol";
import {ERC20Upgradeable} from "@openzeppelin-upgradeable/token/ERC20/ERC20Upgradeable.sol";
import {IERC20Metadata} from "@openzeppelin-contracts/token/ERC20/extensions/IERC20Metadata.sol";
import {PausableUpgradeable} from "@openzeppelin-upgradeable/utils/PausableUpgradeable.sol";
import {Address} from "@openzeppelin-contracts/utils/Address.sol";
import {EnumerableSet} from "@openzeppelin-contracts/utils/structs/EnumerableSet.sol";
import {Ownable} from "@openzeppelin-contracts/access/Ownable.sol";
import {SafeCast} from "@openzeppelin-contracts/utils/math/SafeCast.sol";

// ─────────────────────────────────────────────────────────────────────────────
// Protocol-facing interfaces
// ─────────────────────────────────────────────────────────────────────────────
import {IContractId} from "../interface/IContractId.sol";
import {IUpgradeableVault} from "../interface/IUpgradeableVault.sol";
import {IConcreteStandardVaultImpl} from "../interface/IConcreteStandardVaultImpl.sol";
import {IDefaultEnforcedVaultConfiguration} from "../interface/IDefaultEnforcedVaultConfiguration.sol";
import {IHurdleRateOracle} from "../interface/IHurdleRateOracle.sol";
import {IStrategyTemplate} from "../interface/IStrategyTemplate.sol";

// ─────────────────────────────────────────────────────────────────────────────
// Internal modules/contracts
// ─────────────────────────────────────────────────────────────────────────────
import {IAllocateModule} from "../module/AllocateModule.sol";
import {UpgradeableVault} from "../common/UpgradeableVault.sol";

// ─────────────────────────────────────────────────────────────────────────────
// Internal libraries
// ─────────────────────────────────────────────────────────────────────────────
import {ConcreteV2ConversionLib as ConversionLib} from "../lib/Conversion.sol";
import {StandardVaultHelperLib} from "../lib/StandardVaultHelperLib.sol";
import {Hooks, HooksLibV2 as HooksLib} from "../lib/Hooks.sol";
import {IHook} from "../interface/IHook.sol";
import {ConcreteV2RolesLib as RolesLib} from "../lib/Roles.sol";
import {ERC20UpgradeableWithTransferHook} from "../common/ERC20UpgradeableWithTransferHook.sol";
import {StateInitLib} from "../lib/StateInitLib.sol";
import {StateSetterLib} from "../lib/StateSetterLib.sol";
import {ContractIdEncoding} from "../lib/ContractIdEncoding.sol";
import {WithdrawLib} from "../lib/WithdrawLib.sol";
import {YieldAccrualLib} from "../lib/YieldAccrualLib.sol";

// ─────────────────────────────────────────────────────────────────────────────
// Storage layout libraries
// ─────────────────────────────────────────────────────────────────────────────
import {
    ConcreteCachedVaultStateStorageLib as CachedVaultStateLib
} from "../lib/storage/ConcreteCachedVaultStateStorageLib.sol";
import {ConcreteStandardVaultImplStorageLib as SVLib} from "../lib/storage/ConcreteStandardVaultImplStorageLib.sol";
import {RoleMigrationLib} from "../lib/RoleMigrationLib.sol";

contract ConcreteStandardVaultImpl is
    IContractId,
    ERC20UpgradeableWithTransferHook,
    ERC4626Upgradeable,
    UpgradeableVault,
    AccessControlEnumerableUpgradeable,
    PausableUpgradeable,
    IConcreteStandardVaultImpl
{
    using Address for address;
    using EnumerableSet for EnumerableSet.AddressSet;
    using SafeCast for uint256;
    using ConversionLib for uint256;

    using HooksLib for Hooks;

    event HooksSet(Hooks hooks);

    function ID() public pure virtual override(IContractId, UpgradeableVault) returns (uint64) {
        // contractType=0 (vault), flavour=1 (standard vault), version=1.4.0 (major=1, minor=4, patch=0)
        return uint64(110400); //ContractIdEncoding.encode(0, 1, 1, 04, 00);
    }

    /// @dev Modifier to accrue yield before function execution
    modifier withYieldAccrual() {
        _accrueYield();
        _;
    }
    /**
     * @dev Constructor
     * @param factory The address of the factory
     */

    constructor(address factory) UpgradeableVault(factory) {}

    /// @inheritdoc IConcreteStandardVaultImpl
    function allocate(bytes calldata data)
        public
        virtual
        whenNotPaused
        nonReentrant
        onlyRole(RolesLib.ALLOCATOR)
        withYieldAccrual
    {
        // delegatecall allocate module
        allocateModule().functionDelegateCall(abi.encodeWithSelector(IAllocateModule.allocateFunds.selector, data));
        require(IERC20(asset()).balanceOf(address(this)) >= _lockedAssets(), InsufficientBalance());
    }

    /// @inheritdoc IConcreteStandardVaultImpl
    function accrueYield() external virtual whenNotPaused nonReentrant onlyRole(RolesLib.ALLOCATOR) {
        _accrueYield();
    }

    /// @inheritdoc IConcreteStandardVaultImpl
    function configure(IConcreteStandardVaultImpl.VaultManagerConfig configType, bytes calldata data)
        external
        nonReentrant
        onlyRole(RolesLib.VAULT_MANAGER)
        withYieldAccrual
    {
        StateSetterLib.configure(configType, data);
    }

    /// @inheritdoc IConcreteStandardVaultImpl
    function updateManagementFeeRecipient(address recipient) external nonReentrant withYieldAccrual {
        require(msg.sender == Ownable(factory).owner(), InvalidFactoryOwner());

        StateSetterLib.updateManagementFeeRecipient(recipient);
    }

    /// @inheritdoc IConcreteStandardVaultImpl
    function updatePerformanceFeeRecipient(address recipient) external nonReentrant withYieldAccrual {
        require(msg.sender == Ownable(factory).owner(), InvalidFactoryOwner());

        StateSetterLib.updatePerformanceFeeRecipient(recipient);
    }

    function getHurdleRateOracle() external view returns (IHurdleRateOracle) {
        return SVLib.fetch().hurdleRateOracle;
    }

    /// @inheritdoc IConcreteStandardVaultImpl
    function getFeeConfig()
        external
        view
        override
        returns (
            uint16 currentManagementFee,
            address currentManagementFeeRecipient,
            uint32 currentLastManagementFeeAccrual,
            uint16 currentPerformanceFee,
            address currentPerformanceFeeRecipient
        )
    {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        return (
            $.managementFee,
            $.managementFeeRecipient,
            $.lastManagementFeeAccrual,
            $.performanceFee,
            $.performanceFeeRecipient
        );
    }

    /**
     * @inheritdoc IERC4626
     */
    function deposit(uint256 assets, address receiver)
        public
        virtual
        override(IERC4626, ERC4626Upgradeable)
        whenNotPaused
        nonReentrant
        withYieldAccrual
        returns (uint256)
    {
        require(receiver != address(0), InvalidReceiver());

        Hooks memory h = SVLib.fetch().hooks;
        uint256 totalAssetsBeforeDeposit = cachedTotalAssets();

        uint256 shares = assets.calcConvertToShares(totalSupply(), totalAssetsBeforeDeposit, Math.Rounding.Floor, true);

        // invoke pre-deposit hook if enabled
        if (h.checkIsValid(HooksLib.PRE_DEPOSIT)) {
            h.preDeposit(msg.sender, assets, shares, receiver, totalAssetsBeforeDeposit);
        }

        // Verify deposit limits using cachedTotalAssets (after yield accrual)
        uint256 maxAssets = _maxDeposit(receiver, totalAssetsBeforeDeposit);
        if (assets > maxAssets) {
            revert ERC4626ExceededMaxDeposit(receiver, assets, maxAssets);
        }

        _deposit(msg.sender, receiver, assets, shares);

        // invoke post-deposit hook if enabled
        if (h.checkIsValid(HooksLib.POST_DEPOSIT)) {
            h.postDeposit(msg.sender, assets, shares, receiver, cachedTotalAssets());
        }

        return shares;
    }

    /**
     * @inheritdoc IERC4626
     */
    function mint(uint256 shares, address receiver)
        public
        virtual
        override(IERC4626, ERC4626Upgradeable)
        whenNotPaused
        nonReentrant
        withYieldAccrual
        returns (uint256)
    {
        require(receiver != address(0), InvalidReceiver());

        Hooks memory h = SVLib.fetch().hooks;
        uint256 totalAssetsBeforeDeposit = cachedTotalAssets();
        uint256 totalSupply_ = totalSupply();

        uint256 assets = shares.calcConvertToAssets(totalSupply_, totalAssetsBeforeDeposit, Math.Rounding.Ceil, true);

        // invoke pre-mint hook if enabled
        if (h.checkIsValid(HooksLib.PRE_MINT)) {
            h.preMint(msg.sender, assets, shares, receiver, totalAssetsBeforeDeposit);
        }

        uint256 maxShares = _maxMint(receiver, totalAssetsBeforeDeposit, totalSupply_);
        if (shares > maxShares) {
            revert ERC4626ExceededMaxMint(receiver, shares, maxShares);
        }

        _deposit(msg.sender, receiver, assets, shares);

        // invoke post-mint hook if enabled
        if (h.checkIsValid(HooksLib.POST_MINT)) {
            h.postMint(msg.sender, assets, shares, receiver, cachedTotalAssets());
        }

        return assets;
    }

    /**
     * @inheritdoc IERC4626
     */
    function withdraw(uint256 assets, address receiver, address owner)
        public
        virtual
        override(IERC4626, ERC4626Upgradeable)
        whenNotPaused
        nonReentrant
        withYieldAccrual
        returns (uint256)
    {
        require(receiver != address(0), InvalidReceiver());

        Hooks memory h = SVLib.fetch().hooks;
        uint256 totalAssetsBeforeWithdrawal = cachedTotalAssets();

        uint256 totalSupply_ = totalSupply();
        uint256 shares = assets.calcConvertToShares(totalSupply_, totalAssetsBeforeWithdrawal, Math.Rounding.Ceil, true);

        if (h.checkIsValid(HooksLib.PRE_WITHDRAW)) {
            h.preWithdraw(msg.sender, assets, shares, receiver, owner, totalAssetsBeforeWithdrawal);
        }
        // Optimistic maxAssets that does not account for the withdrawals from strategies, this is to avoid the need to call _simulateWithdraw().
        // If maxAssets is greater than the actual withdrawable amount, _executeWithdraw() will revert.
        uint256 maxAssets = _maxWithdraw(owner, totalAssetsBeforeWithdrawal, totalSupply_);

        if (assets > maxAssets || _executeWithdraw(msg.sender, receiver, owner, assets, shares) < assets) {
            revert ERC4626ExceededMaxWithdraw(owner, assets, maxAssets);
        }

        uint256 minWithdrawAmount = SVLib.fetch().minTxWithdrawAmount;
        require(assets >= minWithdrawAmount, MinimumWithdrawAmountNotMet(msg.sender, assets, minWithdrawAmount));
        // invoke post-withdraw hook if enabled
        if (h.checkIsValid(HooksLib.POST_WITHDRAW)) {
            h.postWithdraw(msg.sender, assets, shares, receiver, cachedTotalAssets());
        }

        return shares;
    }

    /**
     * @inheritdoc IERC4626
     */
    function redeem(uint256 shares, address receiver, address owner)
        public
        virtual
        override(IERC4626, ERC4626Upgradeable)
        whenNotPaused
        nonReentrant
        withYieldAccrual
        returns (uint256)
    {
        require(receiver != address(0), InvalidReceiver());

        Hooks memory h = SVLib.fetch().hooks;
        uint256 totalAssetsBeforeWithdrawal = cachedTotalAssets();
        uint256 totalSupply_ = totalSupply();

        uint256 assets =
            shares.calcConvertToAssets(totalSupply_, totalAssetsBeforeWithdrawal, Math.Rounding.Floor, true);

        // invoke pre-redeem hook if enabled
        if (h.checkIsValid(HooksLib.PRE_REDEEM)) {
            h.preRedeem(msg.sender, assets, shares, receiver, owner, totalAssetsBeforeWithdrawal);
        }

        // Optimistic maxShares that does not account for the withdrawals from strategies, this is to avoid the need to call _simulateWithdraw().
        // If maxShares is greater than the actual redeemable amount, _executeWithdraw() will revert.
        uint256 maxShares = _maxRedeem(owner, totalAssetsBeforeWithdrawal, totalSupply_);

        if (shares > maxShares || _executeWithdraw(msg.sender, receiver, owner, assets, shares) < assets) {
            revert ERC4626ExceededMaxRedeem(owner, shares, maxShares);
        }

        uint256 minWithdrawAmount = SVLib.fetch().minTxWithdrawAmount;
        require(assets >= minWithdrawAmount, MinimumWithdrawAmountNotMet(msg.sender, assets, minWithdrawAmount));

        // invoke post-redeem hook if enabled
        if (h.checkIsValid(HooksLib.POST_REDEEM)) {
            h.postRedeem(msg.sender, assets, shares, receiver, cachedTotalAssets());
        }

        return assets;
    }

    /**
     * @inheritdoc IConcreteStandardVaultImpl
     */
    function setHooks(Hooks memory hooks) external virtual whenNotPaused nonReentrant onlyRole(RolesLib.HOOK_MANAGER) {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        if (hooks.target != address(0) && IHook(hooks.target).vault() != address(this)) {
            revert IConcreteStandardVaultImpl.InvalidHookVault(hooks.target);
        }

        $.hooks = hooks;

        emit HooksSet(hooks);
    }

    /**
     * @inheritdoc IConcreteStandardVaultImpl
     */
    function getHooks() external view returns (Hooks memory) {
        return SVLib.fetch().hooks;
    }

    /**
     * @notice overwrites the deallocation order from strategies;
     */
    function setDeallocationOrder(address[] calldata order)
        external
        virtual
        whenNotPaused
        nonReentrant
        onlyRole(RolesLib.ALLOCATOR)
    {
        StateSetterLib.setDeallocationOrder(order);
    }

    /**
     * @inheritdoc IConcreteStandardVaultImpl
     */
    function addStrategy(address strategy)
        public
        virtual
        whenNotPaused
        nonReentrant
        onlyRole(RolesLib.STRATEGY_MANAGER)
    {
        require(IStrategyTemplate(strategy).asset() == asset(), InvalidStrategyAsset());
        require(IStrategyTemplate(strategy).getVault() == address(this), InvalidStrategyVault());
        StateSetterLib.addStrategy(strategy);
    }

    /**
     * @inheritdoc IConcreteStandardVaultImpl
     */
    function removeStrategy(address strategy)
        public
        virtual
        whenNotPaused
        nonReentrant
        onlyRole(RolesLib.STRATEGY_MANAGER)
    {
        StateSetterLib.removeStrategy(strategy);
    }

    /**
     * @inheritdoc IConcreteStandardVaultImpl
     */
    function toggleStrategyStatus(address strategy)
        public
        virtual
        whenNotPaused
        nonReentrant
        onlyRole(RolesLib.STRATEGY_MANAGER)
    {
        StateSetterLib.toggleStrategyStatus(strategy);
    }

    /**
     * @notice Returns the deallocation order from strategies.
     */
    function getDeallocationOrder() external view returns (address[] memory order) {
        return SVLib.fetch().deallocationOrder;
    }

    /**
     * @dev See {IERC4626-previewDeposit}.
     */
    function previewDeposit(uint256 assets)
        public
        view
        virtual
        override(IERC4626, ERC4626Upgradeable)
        returns (uint256)
    {
        (uint256 totalAssetsPreview, uint256 totalSupply) = _previewAccrueYieldAndFees();
        return assets.calcConvertToShares(totalSupply, totalAssetsPreview, Math.Rounding.Floor, false);
    }

    /**
     * @inheritdoc IERC4626
     */
    function previewMint(uint256 shares) public view virtual override(IERC4626, ERC4626Upgradeable) returns (uint256) {
        (uint256 totalAssetsPreview, uint256 totalSupply) = _previewAccrueYieldAndFees();

        return shares.calcConvertToAssets(totalSupply, totalAssetsPreview, Math.Rounding.Ceil, false);
    }

    /**
     * @inheritdoc IERC4626
     */
    function previewWithdraw(uint256 assets)
        public
        view
        virtual
        override(IERC4626, ERC4626Upgradeable)
        returns (uint256)
    {
        (uint256 totalAssetsPreview, uint256 totalSupply) = _previewAccrueYieldAndFees();

        return assets.calcConvertToShares(totalSupply, totalAssetsPreview, Math.Rounding.Ceil, false);
    }

    /**
     * @inheritdoc IERC4626
     */
    function previewRedeem(uint256 shares)
        public
        view
        virtual
        override(IERC4626, ERC4626Upgradeable)
        returns (uint256)
    {
        (uint256 totalAssetsPreview, uint256 totalSupply) = _previewAccrueYieldAndFees();

        return shares.calcConvertToAssets(totalSupply, totalAssetsPreview, Math.Rounding.Floor, false);
    }

    /**
     * @inheritdoc IConcreteStandardVaultImpl
     */
    function previewAccrueYield() public view virtual returns (uint256, uint256) {
        return _previewAccrueYieldAndFees();
    }

    /**
     * @inheritdoc IERC4626
     */
    function totalAssets()
        public
        view
        virtual
        override(IERC4626, ERC4626Upgradeable)
        returns (uint256 totalAssetsWithYield)
    {
        (totalAssetsWithYield,) = _previewAccrueYieldAndFees();
    }

    function maxDeposit(address receiver) public view virtual override(IERC4626, ERC4626Upgradeable) returns (uint256) {
        if (paused()) return 0;
        (uint256 totalAssetsPreview,) = _previewAccrueYieldAndFees();
        return _maxDeposit(receiver, totalAssetsPreview);
    }

    function maxMint(address receiver) public view virtual override(IERC4626, ERC4626Upgradeable) returns (uint256) {
        if (paused()) return 0;
        (uint256 totalAssetsPreview, uint256 totalSupplyPreview) = _previewAccrueYieldAndFees();
        return _maxMint(receiver, totalAssetsPreview, totalSupplyPreview);
    }

    function _maxMint(address, uint256 totalAssets_, uint256 totalSupply_) internal view virtual returns (uint256) {
        return StandardVaultHelperLib.calcMaxMint(
            SVLib.fetch().maxGlobalDepositAmount, SVLib.fetch().minTxDepositAmount, totalAssets_, totalSupply_
        );
    }

    function _maxDeposit(address, uint256 totalAssets_) internal view virtual returns (uint256) {
        return StandardVaultHelperLib.calcMaxDeposit(
            SVLib.fetch().maxGlobalDepositAmount, SVLib.fetch().minTxDepositAmount, totalAssets_
        );
    }

    /**
     * @inheritdoc IERC4626
     */
    function maxRedeem(address owner) public view virtual override(IERC4626, ERC4626Upgradeable) returns (uint256) {
        if (paused()) return 0;
        (uint256 expectedTotalAssets, uint256 expectedTotalSupply) = _previewAccrueYieldAndFees();

        // Get max shares using the same logic as redeem() - uses corrected calcMaxRedeem (avoids double rounding)
        uint256 maxShares = _maxRedeem(owner, expectedTotalAssets, expectedTotalSupply);

        // Get available liquidity (idle + withdrawable from strategies)
        // This matches the test expectation: convertToShares(min(userAssets, availableLiquidity))
        uint256 availableLiquidity = _simulateWithdraw(type(uint256).max);

        // Convert available liquidity to shares
        uint256 availableShares = availableLiquidity.calcConvertToShares(
            expectedTotalSupply, expectedTotalAssets, Math.Rounding.Floor, false
        );

        // Return the minimum to ensure maxRedeem never reports more than what redeem() would accept
        // This prevents inconsistency where maxRedeem reports more shares than redeem accepts
        return availableShares > maxShares ? maxShares : availableShares;
    }

    function _maxRedeem(address owner, uint256 totalAssets_, uint256 totalSupply_)
        internal
        view
        virtual
        returns (uint256)
    {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();
        return StandardVaultHelperLib.calcMaxRedeem(
            balanceOf(owner), $.maxTxWithdrawAmount, $.minTxWithdrawAmount, totalAssets_, totalSupply_
        );
    }

    /**
     * @inheritdoc IERC4626
     */
    function maxWithdraw(address owner)
        public
        view
        virtual
        override(IERC4626, ERC4626Upgradeable)
        returns (uint256 maxAssets)
    {
        if (paused()) return 0;
        (maxAssets,,) = _maxWithdrawWithSimulation(owner);
    }

    function _maxWithdraw(address owner, uint256 totalAssets_, uint256 totalSupply_)
        internal
        view
        virtual
        returns (uint256)
    {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();
        return StandardVaultHelperLib.calcMaxWithdraw(
            balanceOf(owner), $.maxTxWithdrawAmount, $.minTxWithdrawAmount, totalAssets_, totalSupply_
        );
    }

    /**
     * @inheritdoc IConcreteStandardVaultImpl
     */
    function getStrategyData(address strategy) public view returns (StrategyData memory) {
        return SVLib.fetch().strategyData[strategy];
    }

    /**
     * @inheritdoc IConcreteStandardVaultImpl
     */
    function getStrategies() public view returns (address[] memory) {
        return SVLib.fetch().strategies.values();
    }

    /**
     * @inheritdoc IConcreteStandardVaultImpl
     */
    function allocateModule() public view returns (address) {
        return SVLib.fetch().allocateModule;
    }

    function getDepositLimits() public view returns (uint256, uint256) {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();
        return ($.maxGlobalDepositAmount, $.minTxDepositAmount);
    }

    /// @inheritdoc IConcreteStandardVaultImpl
    function getWithdrawLimits() public view returns (uint256, uint256) {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();
        return ($.maxTxWithdrawAmount, $.minTxWithdrawAmount);
    }

    /// @inheritdoc IConcreteStandardVaultImpl
    function getTotalAllocated() public view returns (uint256) {
        return StandardVaultHelperLib.calcTotalAllocated();
    }

    function cachedTotalAssets() public view returns (uint256) {
        return CachedVaultStateLib.fetch().cachedTotalAssets;
    }

    /**
     * @dev Initialization function that will be called when a proxy vault is deployed through `ConcreteFactory`.
     */
    function _initialize(
        uint64,
        /*initialVersion*/
        address,
        /*owner*/
        IDefaultEnforcedVaultConfiguration.DefaultEnforcedVaultConfiguration calldata defaultEnforcedVaultConfig,
        bytes memory data
    ) internal virtual override {
        (
            address allocateModuleAddr,
            address asset,
            address initialVaultManager,
            string memory name,
            string memory symbol
        ) = abi.decode(data, (address, address, address, string, string));

        require(allocateModuleAddr != address(0), InvalidAllocateModule());
        require(asset != address(0), InvalidAsset());
        require(initialVaultManager != address(0), InvalidInitialVaultManager());

        require(bytes(name).length > 0, InvalidName());
        require(bytes(symbol).length > 0, InvalidSymbol());

        __ERC20_init_unchained(name, symbol);
        __ERC4626_init_unchained(IERC20(asset));
        __AccessControlEnumerable_init_unchained();
        __Pausable_init_unchained();

        StateSetterLib.updateManagementFeeRecipient(defaultEnforcedVaultConfig.defaultManagementFeeRecipient);
        StateSetterLib.updatePerformanceFeeRecipient(defaultEnforcedVaultConfig.defaultPerformanceFeeRecipient);

        StateInitLib.stateInitStandardVaultImpl(allocateModuleAddr, initialVaultManager, msg.sender);
    }

    /**
     * @dev Upgrade function that will be called when a proxy vault upgrades to this implementation
     */
    function _upgrade(
        uint64,
        /* newVersion */
        bytes calldata /* data */
    )
        internal
        virtual
        override
        whenNotPaused
    {
        // Migrate legacy admin roles to ROLE_ADMIN
        RoleMigrationLib.migrateLegacyAdminRoles(msg.sender);
    }

    /**
     * @notice Pauses the vault, preventing critical operations
     * @dev Only PAUSER role can pause the vault
     */
    function pause() external onlyRole(RolesLib.PAUSER) {
        _pause();
    }

    /**
     * @notice Unpauses the vault, allowing critical operations
     * @dev Only PAUSER role can unpause the vault
     */
    function unpause() external onlyRole(RolesLib.PAUSER) {
        _unpause();
    }

    /**
     * @dev Internal function that executes the yield accrual operation across all active strategies.
     */
    function _accrueYield() internal virtual returns (uint256) {
        return YieldAccrualLib.accrueYield();
    }

    /**
     * @dev Internal function that handles withdrawal operations, including strategy deallocation when needed.
     * @dev This function implements the core withdrawal logic for the vault, automatically managing
     *      fund retrieval from both idle vault balance and allocated strategies to fulfill withdrawal requests.
     * @dev Withdrawal Process:
     *      1. First attempts to use idle funds (assets sitting in the vault contract)
     *      2. If idle funds are insufficient, iterates through active strategies to deallocate funds
     *      3. For each strategy, respects the strategy's maxWithdraw() limit
     *      4. Updates strategy allocation accounting after successful deallocations
     *      5. Executes the final withdrawal by burning shares, updating cached total assets, and transferring assets to the receiver
     * @dev Requirements:
     *      - Combined idle funds and strategy liquidity must be sufficient for withdrawal amount
     *      - All deallocated strategies must be in Active status
     *      - Strategy onWithdraw() calls must succeed and return expected amounts
     * @param caller The address that initiated the withdrawal (for access control)
     * @param receiver The address that will receive the withdrawn assets
     * @param owner The address whose shares are being burned for the withdrawal
     * @param assets The amount of assets to withdraw from the vault
     * @param shares The amount of shares to burn in exchange for the assets
     */
    function _executeWithdraw(address caller, address receiver, address owner, uint256 assets, uint256 shares)
        internal
        virtual
        returns (uint256)
    {
        uint256 totalWithdrawableAmount = WithdrawLib.executeWithdrawFromStrategies(asset(), assets, _lockedAssets());

        if (totalWithdrawableAmount < assets) {
            revert IConcreteStandardVaultImpl.InsufficientWithdrawableAssets(assets, totalWithdrawableAmount);
        }

        _withdraw(caller, receiver, owner, assets, shares);

        return totalWithdrawableAmount;
    }

    function _withdraw(address caller, address receiver, address owner, uint256 assets, uint256 shares)
        internal
        virtual
        override
    {
        if (caller != owner) {
            _spendAllowance(owner, caller, shares);
        }

        // If asset() is ERC-777, `transfer` can trigger a reentrancy AFTER the transfer happens through the
        // `tokensReceived` hook. On the other hand, the `tokensToSend` hook, that is triggered before the transfer,
        // calls the vault, which is assumed not malicious.
        //
        // Conclusion: we need to do the transfer after the burn so that any reentrancy would happen after the
        // shares are burned and after the assets are transferred, which is a valid state.

        CachedVaultStateLib.fetch().cachedTotalAssets -= assets;
        _burn(owner, shares);
        SafeERC20.safeTransfer(IERC20(asset()), receiver, assets);

        emit Withdraw(caller, receiver, owner, assets, shares);
    }

    function decimals()
        public
        view
        virtual
        override(ERC4626Upgradeable, ERC20Upgradeable, IERC20Metadata)
        returns (uint8)
    {
        return ERC4626Upgradeable.decimals();
    }

    function _update(address from, address to, uint256 value)
        internal
        virtual
        override(ERC20UpgradeableWithTransferHook, ERC20Upgradeable)
    {
        ERC20UpgradeableWithTransferHook._update(from, to, value);
    }

    /**
     * @dev Deposit/mint common workflow.
     */
    function _deposit(address caller, address receiver, uint256 assets, uint256 shares) internal virtual override {
        // If asset() is ERC-777, `transferFrom` can trigger a reentrancy BEFORE the transfer happens through the
        // `tokensToSend` hook. On the other hand, the `tokenReceived` hook, that is triggered after the transfer,
        // calls the vault, which is assumed not malicious.
        //
        // Conclusion: we need to do the transfer before we mint so that any reentrancy would happen before the
        // assets are transferred and before the shares are minted, which is a valid state.
        // slither-disable-next-line reentrancy-no-eth
        uint256 minDepositAmount = SVLib.fetch().minTxDepositAmount;
        require(assets >= minDepositAmount, MinimumDepositAmountNotMet(msg.sender, assets, minDepositAmount));

        SafeERC20.safeTransferFrom(IERC20(asset()), caller, address(this), assets);
        CachedVaultStateLib.fetch().cachedTotalAssets += assets;
        _mint(receiver, shares);

        emit Deposit(caller, receiver, assets, shares);
    }

    /**
     * @dev Simulates the yield accrual operation across all strategies including fees.
     * @return totalAssets The projected total assets after yield accrual (current + yield - losses)
     * @return totalSupply The projected total supply after yield accrual (current + management fee shares + performance fee shares)
     */
    function _previewAccrueYieldAndFees() internal view virtual returns (uint256, uint256) {
        return YieldAccrualLib.previewAccrueYieldAndFees(cachedTotalAssets());
    }

    /**
     * @dev Calculates total positive and negative yield across all strategies.
     * @return totalPositiveYield The sum of all positive yields from strategies
     * @return totalNegativeYield The sum of all losses from strategies
     */
    function _previewYieldNoFees()
        internal
        view
        virtual
        returns (uint256 totalPositiveYield, uint256 totalNegativeYield)
    {
        return YieldAccrualLib.previewYieldNoFees();
    }

    /**
     * @dev Internal function to calculate the maximum amount of assets that can be withdrawn by an owner.
     * @dev The calculation considers:
     *      - Owner's current share balance converted to equivalent assets
     *      - Current vault liquidity (idle assets + withdrawable amounts from strategies)
     *      - Strategy withdrawal limitations and availability
     * @param owner The address of the account for which to calculate maximum withdrawal.
     * @return The maximum amount of assets that can actually be withdrawn by the owner,
     *         considering both ownership rights and liquidity constraints.
     * @return totalAssets The total amount of assets in the vault after previewing the yield accrual
     */
    function _maxWithdrawWithSimulation(address owner) internal view virtual returns (uint256, uint256, uint256) {
        (uint256 totalAssetsPreview, uint256 totalSupplyPreview) = _previewAccrueYieldAndFees();

        uint256 maxAssets = _maxWithdraw(owner, totalAssetsPreview, totalSupplyPreview);

        return (_simulateWithdraw(maxAssets), totalAssetsPreview, totalSupplyPreview);
    }

    /// @dev Simulate withdrawing an amount of assets from the vault.
    /// @param requestedAssets Amount of assets to withdraw.
    /// @return Amount of assets filled.
    function _simulateWithdraw(uint256 requestedAssets) internal view virtual returns (uint256) {
        return WithdrawLib.simulateWithdraw(asset(), requestedAssets, _lockedAssets());
    }

    /**
     * @dev Internal function used only by the public convertToShares() function.
     * @dev This function returns share amounts NOT inclusive of management or performance fees,
     *      as required by the ERC4626 specification. The convertToShares() function should return
     *      the theoretical share amount for a given asset amount without considering fee deductions.
     * @param assets The amount of assets to convert to shares
     * @param rounding The rounding direction for the conversion
     * @return The equivalent amount of shares, NOT inclusive of fees
     */
    function _convertToShares(uint256 assets, Math.Rounding rounding) internal view virtual override returns (uint256) {
        (uint256 totalPositiveYield, uint256 totalNegativeYield) = _previewYieldNoFees();
        return assets.calcConvertToShares(
            totalSupply(), cachedTotalAssets() + totalPositiveYield - totalNegativeYield, rounding, false
        );
    }

    /**
     * @dev Internal function used only by the public convertToAssets() function.
     * @dev This function returns asset amounts NOT inclusive of management or performance fees,
     *      as required by the ERC4626 specification. The convertToAssets() function should return
     *      the theoretical asset amount for a given share amount without considering fee deductions.
     * @param shares The amount of shares to convert to assets
     * @param rounding The rounding direction for the conversion
     * @return The equivalent amount of assets, NOT inclusive of fees
     */
    function _convertToAssets(uint256 shares, Math.Rounding rounding) internal view virtual override returns (uint256) {
        (uint256 totalPositiveYield, uint256 totalNegativeYield) = _previewYieldNoFees();
        return shares.calcConvertToAssets(
            totalSupply(), cachedTotalAssets() + totalPositiveYield - totalNegativeYield, rounding, false
        );
    }

    function _lockedAssets() internal view virtual returns (uint256) {
        return 0;
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts-upgradeable/access/extensions/AccessControlEnumerableUpgradeable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (access/extensions/AccessControlEnumerable.sol)

pragma solidity ^0.8.20;

import {IAccessControlEnumerable} from "@openzeppelin/contracts/access/extensions/IAccessControlEnumerable.sol";
import {AccessControlUpgradeable} from "../AccessControlUpgradeable.sol";
import {EnumerableSet} from "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";
import {Initializable} from "../../proxy/utils/Initializable.sol";

/**
 * @dev Extension of {AccessControl} that allows enumerating the members of each role.
 */
abstract contract AccessControlEnumerableUpgradeable is Initializable, IAccessControlEnumerable, AccessControlUpgradeable {
    using EnumerableSet for EnumerableSet.AddressSet;

    /// @custom:storage-location erc7201:openzeppelin.storage.AccessControlEnumerable
    struct AccessControlEnumerableStorage {
        mapping(bytes32 role => EnumerableSet.AddressSet) _roleMembers;
    }

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.AccessControlEnumerable")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant AccessControlEnumerableStorageLocation = 0xc1f6fe24621ce81ec5827caf0253cadb74709b061630e6b55e82371705932000;

    function _getAccessControlEnumerableStorage() private pure returns (AccessControlEnumerableStorage storage $) {
        assembly {
            $.slot := AccessControlEnumerableStorageLocation
        }
    }

    function __AccessControlEnumerable_init() internal onlyInitializing {
    }

    function __AccessControlEnumerable_init_unchained() internal onlyInitializing {
    }
    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId) public view virtual override returns (bool) {
        return interfaceId == type(IAccessControlEnumerable).interfaceId || super.supportsInterface(interfaceId);
    }

    /**
     * @dev Returns one of the accounts that have `role`. `index` must be a
     * value between 0 and {getRoleMemberCount}, non-inclusive.
     *
     * Role bearers are not sorted in any particular way, and their ordering may
     * change at any point.
     *
     * WARNING: When using {getRoleMember} and {getRoleMemberCount}, make sure
     * you perform all queries on the same block. See the following
     * https://forum.openzeppelin.com/t/iterating-over-elements-on-enumerableset-in-openzeppelin-contracts/2296[forum post]
     * for more information.
     */
    function getRoleMember(bytes32 role, uint256 index) public view virtual returns (address) {
        AccessControlEnumerableStorage storage $ = _getAccessControlEnumerableStorage();
        return $._roleMembers[role].at(index);
    }

    /**
     * @dev Returns the number of accounts that have `role`. Can be used
     * together with {getRoleMember} to enumerate all bearers of a role.
     */
    function getRoleMemberCount(bytes32 role) public view virtual returns (uint256) {
        AccessControlEnumerableStorage storage $ = _getAccessControlEnumerableStorage();
        return $._roleMembers[role].length();
    }

    /**
     * @dev Return all accounts that have `role`
     *
     * WARNING: This operation will copy the entire storage to memory, which can be quite expensive. This is designed
     * to mostly be used by view accessors that are queried without any gas fees. Developers should keep in mind that
     * this function has an unbounded cost, and using it as part of a state-changing function may render the function
     * uncallable if the set grows to a point where copying to memory consumes too much gas to fit in a block.
     */
    function getRoleMembers(bytes32 role) public view virtual returns (address[] memory) {
        AccessControlEnumerableStorage storage $ = _getAccessControlEnumerableStorage();
        return $._roleMembers[role].values();
    }

    /**
     * @dev Overload {AccessControl-_grantRole} to track enumerable memberships
     */
    function _grantRole(bytes32 role, address account) internal virtual override returns (bool) {
        AccessControlEnumerableStorage storage $ = _getAccessControlEnumerableStorage();
        bool granted = super._grantRole(role, account);
        if (granted) {
            $._roleMembers[role].add(account);
        }
        return granted;
    }

    /**
     * @dev Overload {AccessControl-_revokeRole} to track enumerable memberships
     */
    function _revokeRole(bytes32 role, address account) internal virtual override returns (bool) {
        AccessControlEnumerableStorage storage $ = _getAccessControlEnumerableStorage();
        bool revoked = super._revokeRole(role, account);
        if (revoked) {
            $._roleMembers[role].remove(account);
        }
        return revoked;
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts-upgradeable/token/ERC20/extensions/ERC4626Upgradeable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (token/ERC20/extensions/ERC4626.sol)

pragma solidity ^0.8.20;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {IERC20Metadata} from "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import {ERC20Upgradeable} from "../ERC20Upgradeable.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {IERC4626} from "@openzeppelin/contracts/interfaces/IERC4626.sol";
import {Math} from "@openzeppelin/contracts/utils/math/Math.sol";
import {Initializable} from "../../../proxy/utils/Initializable.sol";

/**
 * @dev Implementation of the ERC-4626 "Tokenized Vault Standard" as defined in
 * https://eips.ethereum.org/EIPS/eip-4626[ERC-4626].
 *
 * This extension allows the minting and burning of "shares" (represented using the ERC-20 inheritance) in exchange for
 * underlying "assets" through standardized {deposit}, {mint}, {redeem} and {burn} workflows. This contract extends
 * the ERC-20 standard. Any additional extensions included along it would affect the "shares" token represented by this
 * contract and not the "assets" token which is an independent contract.
 *
 * [CAUTION]
 * ====
 * In empty (or nearly empty) ERC-4626 vaults, deposits are at high risk of being stolen through frontrunning
 * with a "donation" to the vault that inflates the price of a share. This is variously known as a donation or inflation
 * attack and is essentially a problem of slippage. Vault deployers can protect against this attack by making an initial
 * deposit of a non-trivial amount of the asset, such that price manipulation becomes infeasible. Withdrawals may
 * similarly be affected by slippage. Users can protect against this attack as well as unexpected slippage in general by
 * verifying the amount received is as expected, using a wrapper that performs these checks such as
 * https://github.com/fei-protocol/ERC4626#erc4626router-and-base[ERC4626Router].
 *
 * Since v4.9, this implementation introduces configurable virtual assets and shares to help developers mitigate that risk.
 * The `_decimalsOffset()` corresponds to an offset in the decimal representation between the underlying asset's decimals
 * and the vault decimals. This offset also determines the rate of virtual shares to virtual assets in the vault, which
 * itself determines the initial exchange rate. While not fully preventing the attack, analysis shows that the default
 * offset (0) makes it non-profitable even if an attacker is able to capture value from multiple user deposits, as a result
 * of the value being captured by the virtual shares (out of the attacker's donation) matching the attacker's expected gains.
 * With a larger offset, the attack becomes orders of magnitude more expensive than it is profitable. More details about the
 * underlying math can be found xref:erc4626.adoc#inflation-attack[here].
 *
 * The drawback of this approach is that the virtual shares do capture (a very small) part of the value being accrued
 * to the vault. Also, if the vault experiences losses, the users try to exit the vault, the virtual shares and assets
 * will cause the first user to exit to experience reduced losses in detriment to the last users that will experience
 * bigger losses. Developers willing to revert back to the pre-v4.9 behavior just need to override the
 * `_convertToShares` and `_convertToAssets` functions.
 *
 * To learn more, check out our xref:ROOT:erc4626.adoc[ERC-4626 guide].
 * ====
 */
abstract contract ERC4626Upgradeable is Initializable, ERC20Upgradeable, IERC4626 {
    using Math for uint256;

    /// @custom:storage-location erc7201:openzeppelin.storage.ERC4626
    struct ERC4626Storage {
        IERC20 _asset;
        uint8 _underlyingDecimals;
    }

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.ERC4626")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant ERC4626StorageLocation = 0x0773e532dfede91f04b12a73d3d2acd361424f41f76b4fb79f090161e36b4e00;

    function _getERC4626Storage() private pure returns (ERC4626Storage storage $) {
        assembly {
            $.slot := ERC4626StorageLocation
        }
    }

    /**
     * @dev Attempted to deposit more assets than the max amount for `receiver`.
     */
    error ERC4626ExceededMaxDeposit(address receiver, uint256 assets, uint256 max);

    /**
     * @dev Attempted to mint more shares than the max amount for `receiver`.
     */
    error ERC4626ExceededMaxMint(address receiver, uint256 shares, uint256 max);

    /**
     * @dev Attempted to withdraw more assets than the max amount for `receiver`.
     */
    error ERC4626ExceededMaxWithdraw(address owner, uint256 assets, uint256 max);

    /**
     * @dev Attempted to redeem more shares than the max amount for `receiver`.
     */
    error ERC4626ExceededMaxRedeem(address owner, uint256 shares, uint256 max);

    /**
     * @dev Set the underlying asset contract. This must be an ERC20-compatible contract (ERC-20 or ERC-777).
     */
    function __ERC4626_init(IERC20 asset_) internal onlyInitializing {
        __ERC4626_init_unchained(asset_);
    }

    function __ERC4626_init_unchained(IERC20 asset_) internal onlyInitializing {
        ERC4626Storage storage $ = _getERC4626Storage();
        (bool success, uint8 assetDecimals) = _tryGetAssetDecimals(asset_);
        $._underlyingDecimals = success ? assetDecimals : 18;
        $._asset = asset_;
    }

    /**
     * @dev Attempts to fetch the asset decimals. A return value of false indicates that the attempt failed in some way.
     */
    function _tryGetAssetDecimals(IERC20 asset_) private view returns (bool ok, uint8 assetDecimals) {
        (bool success, bytes memory encodedDecimals) = address(asset_).staticcall(
            abi.encodeCall(IERC20Metadata.decimals, ())
        );
        if (success && encodedDecimals.length >= 32) {
            uint256 returnedDecimals = abi.decode(encodedDecimals, (uint256));
            if (returnedDecimals <= type(uint8).max) {
                return (true, uint8(returnedDecimals));
            }
        }
        return (false, 0);
    }

    /**
     * @dev Decimals are computed by adding the decimal offset on top of the underlying asset's decimals. This
     * "original" value is cached during construction of the vault contract. If this read operation fails (e.g., the
     * asset has not been created yet), a default of 18 is used to represent the underlying asset's decimals.
     *
     * See {IERC20Metadata-decimals}.
     */
    function decimals() public view virtual override(IERC20Metadata, ERC20Upgradeable) returns (uint8) {
        ERC4626Storage storage $ = _getERC4626Storage();
        return $._underlyingDecimals + _decimalsOffset();
    }

    /** @dev See {IERC4626-asset}. */
    function asset() public view virtual returns (address) {
        ERC4626Storage storage $ = _getERC4626Storage();
        return address($._asset);
    }

    /** @dev See {IERC4626-totalAssets}. */
    function totalAssets() public view virtual returns (uint256) {
        ERC4626Storage storage $ = _getERC4626Storage();
        return $._asset.balanceOf(address(this));
    }

    /** @dev See {IERC4626-convertToShares}. */
    function convertToShares(uint256 assets) public view virtual returns (uint256) {
        return _convertToShares(assets, Math.Rounding.Floor);
    }

    /** @dev See {IERC4626-convertToAssets}. */
    function convertToAssets(uint256 shares) public view virtual returns (uint256) {
        return _convertToAssets(shares, Math.Rounding.Floor);
    }

    /** @dev See {IERC4626-maxDeposit}. */
    function maxDeposit(address) public view virtual returns (uint256) {
        return type(uint256).max;
    }

    /** @dev See {IERC4626-maxMint}. */
    function maxMint(address) public view virtual returns (uint256) {
        return type(uint256).max;
    }

    /** @dev See {IERC4626-maxWithdraw}. */
    function maxWithdraw(address owner) public view virtual returns (uint256) {
        return _convertToAssets(balanceOf(owner), Math.Rounding.Floor);
    }

    /** @dev See {IERC4626-maxRedeem}. */
    function maxRedeem(address owner) public view virtual returns (uint256) {
        return balanceOf(owner);
    }

    /** @dev See {IERC4626-previewDeposit}. */
    function previewDeposit(uint256 assets) public view virtual returns (uint256) {
        return _convertToShares(assets, Math.Rounding.Floor);
    }

    /** @dev See {IERC4626-previewMint}. */
    function previewMint(uint256 shares) public view virtual returns (uint256) {
        return _convertToAssets(shares, Math.Rounding.Ceil);
    }

    /** @dev See {IERC4626-previewWithdraw}. */
    function previewWithdraw(uint256 assets) public view virtual returns (uint256) {
        return _convertToShares(assets, Math.Rounding.Ceil);
    }

    /** @dev See {IERC4626-previewRedeem}. */
    function previewRedeem(uint256 shares) public view virtual returns (uint256) {
        return _convertToAssets(shares, Math.Rounding.Floor);
    }

    /** @dev See {IERC4626-deposit}. */
    function deposit(uint256 assets, address receiver) public virtual returns (uint256) {
        uint256 maxAssets = maxDeposit(receiver);
        if (assets > maxAssets) {
            revert ERC4626ExceededMaxDeposit(receiver, assets, maxAssets);
        }

        uint256 shares = previewDeposit(assets);
        _deposit(_msgSender(), receiver, assets, shares);

        return shares;
    }

    /** @dev See {IERC4626-mint}. */
    function mint(uint256 shares, address receiver) public virtual returns (uint256) {
        uint256 maxShares = maxMint(receiver);
        if (shares > maxShares) {
            revert ERC4626ExceededMaxMint(receiver, shares, maxShares);
        }

        uint256 assets = previewMint(shares);
        _deposit(_msgSender(), receiver, assets, shares);

        return assets;
    }

    /** @dev See {IERC4626-withdraw}. */
    function withdraw(uint256 assets, address receiver, address owner) public virtual returns (uint256) {
        uint256 maxAssets = maxWithdraw(owner);
        if (assets > maxAssets) {
            revert ERC4626ExceededMaxWithdraw(owner, assets, maxAssets);
        }

        uint256 shares = previewWithdraw(assets);
        _withdraw(_msgSender(), receiver, owner, assets, shares);

        return shares;
    }

    /** @dev See {IERC4626-redeem}. */
    function redeem(uint256 shares, address receiver, address owner) public virtual returns (uint256) {
        uint256 maxShares = maxRedeem(owner);
        if (shares > maxShares) {
            revert ERC4626ExceededMaxRedeem(owner, shares, maxShares);
        }

        uint256 assets = previewRedeem(shares);
        _withdraw(_msgSender(), receiver, owner, assets, shares);

        return assets;
    }

    /**
     * @dev Internal conversion function (from assets to shares) with support for rounding direction.
     */
    function _convertToShares(uint256 assets, Math.Rounding rounding) internal view virtual returns (uint256) {
        return assets.mulDiv(totalSupply() + 10 ** _decimalsOffset(), totalAssets() + 1, rounding);
    }

    /**
     * @dev Internal conversion function (from shares to assets) with support for rounding direction.
     */
    function _convertToAssets(uint256 shares, Math.Rounding rounding) internal view virtual returns (uint256) {
        return shares.mulDiv(totalAssets() + 1, totalSupply() + 10 ** _decimalsOffset(), rounding);
    }

    /**
     * @dev Deposit/mint common workflow.
     */
    function _deposit(address caller, address receiver, uint256 assets, uint256 shares) internal virtual {
        ERC4626Storage storage $ = _getERC4626Storage();
        // If _asset is ERC-777, `transferFrom` can trigger a reentrancy BEFORE the transfer happens through the
        // `tokensToSend` hook. On the other hand, the `tokenReceived` hook, that is triggered after the transfer,
        // calls the vault, which is assumed not malicious.
        //
        // Conclusion: we need to do the transfer before we mint so that any reentrancy would happen before the
        // assets are transferred and before the shares are minted, which is a valid state.
        // slither-disable-next-line reentrancy-no-eth
        SafeERC20.safeTransferFrom($._asset, caller, address(this), assets);
        _mint(receiver, shares);

        emit Deposit(caller, receiver, assets, shares);
    }

    /**
     * @dev Withdraw/redeem common workflow.
     */
    function _withdraw(
        address caller,
        address receiver,
        address owner,
        uint256 assets,
        uint256 shares
    ) internal virtual {
        ERC4626Storage storage $ = _getERC4626Storage();
        if (caller != owner) {
            _spendAllowance(owner, caller, shares);
        }

        // If _asset is ERC-777, `transfer` can trigger a reentrancy AFTER the transfer happens through the
        // `tokensReceived` hook. On the other hand, the `tokensToSend` hook, that is triggered before the transfer,
        // calls the vault, which is assumed not malicious.
        //
        // Conclusion: we need to do the transfer after the burn so that any reentrancy would happen after the
        // shares are burned and after the assets are transferred, which is a valid state.
        _burn(owner, shares);
        SafeERC20.safeTransfer($._asset, receiver, assets);

        emit Withdraw(caller, receiver, owner, assets, shares);
    }

    function _decimalsOffset() internal view virtual returns (uint8) {
        return 0;
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts-upgradeable/token/ERC20/ERC20Upgradeable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.2.0) (token/ERC20/ERC20.sol)

pragma solidity ^0.8.20;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {IERC20Metadata} from "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import {ContextUpgradeable} from "../../utils/ContextUpgradeable.sol";
import {IERC20Errors} from "@openzeppelin/contracts/interfaces/draft-IERC6093.sol";
import {Initializable} from "../../proxy/utils/Initializable.sol";

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
abstract contract ERC20Upgradeable is Initializable, ContextUpgradeable, IERC20, IERC20Metadata, IERC20Errors {
    /// @custom:storage-location erc7201:openzeppelin.storage.ERC20
    struct ERC20Storage {
        mapping(address account => uint256) _balances;

        mapping(address account => mapping(address spender => uint256)) _allowances;

        uint256 _totalSupply;

        string _name;
        string _symbol;
    }

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.ERC20")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant ERC20StorageLocation = 0x52c63247e1f47db19d5ce0460030c497f067ca4cebf71ba98eeadabe20bace00;

    function _getERC20Storage() private pure returns (ERC20Storage storage $) {
        assembly {
            $.slot := ERC20StorageLocation
        }
    }

    /**
     * @dev Sets the values for {name} and {symbol}.
     *
     * All two of these values are immutable: they can only be set once during
     * construction.
     */
    function __ERC20_init(string memory name_, string memory symbol_) internal onlyInitializing {
        __ERC20_init_unchained(name_, symbol_);
    }

    function __ERC20_init_unchained(string memory name_, string memory symbol_) internal onlyInitializing {
        ERC20Storage storage $ = _getERC20Storage();
        $._name = name_;
        $._symbol = symbol_;
    }

    /**
     * @dev Returns the name of the token.
     */
    function name() public view virtual returns (string memory) {
        ERC20Storage storage $ = _getERC20Storage();
        return $._name;
    }

    /**
     * @dev Returns the symbol of the token, usually a shorter version of the
     * name.
     */
    function symbol() public view virtual returns (string memory) {
        ERC20Storage storage $ = _getERC20Storage();
        return $._symbol;
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

    /**
     * @dev See {IERC20-totalSupply}.
     */
    function totalSupply() public view virtual returns (uint256) {
        ERC20Storage storage $ = _getERC20Storage();
        return $._totalSupply;
    }

    /**
     * @dev See {IERC20-balanceOf}.
     */
    function balanceOf(address account) public view virtual returns (uint256) {
        ERC20Storage storage $ = _getERC20Storage();
        return $._balances[account];
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

    /**
     * @dev See {IERC20-allowance}.
     */
    function allowance(address owner, address spender) public view virtual returns (uint256) {
        ERC20Storage storage $ = _getERC20Storage();
        return $._allowances[owner][spender];
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
        ERC20Storage storage $ = _getERC20Storage();
        if (from == address(0)) {
            // Overflow check required: The rest of the code assumes that totalSupply never overflows
            $._totalSupply += value;
        } else {
            uint256 fromBalance = $._balances[from];
            if (fromBalance < value) {
                revert ERC20InsufficientBalance(from, fromBalance, value);
            }
            unchecked {
                // Overflow not possible: value <= fromBalance <= totalSupply.
                $._balances[from] = fromBalance - value;
            }
        }

        if (to == address(0)) {
            unchecked {
                // Overflow not possible: value <= totalSupply or value <= fromBalance <= totalSupply.
                $._totalSupply -= value;
            }
        } else {
            unchecked {
                // Overflow not possible: balance + value is at most totalSupply, which we know fits into a uint256.
                $._balances[to] += value;
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
     * @dev Sets `value` as the allowance of `spender` over the `owner` s tokens.
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
        ERC20Storage storage $ = _getERC20Storage();
        if (owner == address(0)) {
            revert ERC20InvalidApprover(address(0));
        }
        if (spender == address(0)) {
            revert ERC20InvalidSpender(address(0));
        }
        $._allowances[owner][spender] = value;
        if (emitEvent) {
            emit Approval(owner, spender, value);
        }
    }

    /**
     * @dev Updates `owner` s allowance for `spender` based on spent `value`.
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


// ===== FILE: node_modules/_openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (token/ERC20/extensions/IERC20Metadata.sol)

pragma solidity ^0.8.20;

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


// ===== FILE: node_modules/_openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (utils/Pausable.sol)

pragma solidity ^0.8.20;

import {ContextUpgradeable} from "../utils/ContextUpgradeable.sol";
import {Initializable} from "../proxy/utils/Initializable.sol";

/**
 * @dev Contract module which allows children to implement an emergency stop
 * mechanism that can be triggered by an authorized account.
 *
 * This module is used through inheritance. It will make available the
 * modifiers `whenNotPaused` and `whenPaused`, which can be applied to
 * the functions of your contract. Note that they will not be pausable by
 * simply including this module, only once the modifiers are put in place.
 */
abstract contract PausableUpgradeable is Initializable, ContextUpgradeable {
    /// @custom:storage-location erc7201:openzeppelin.storage.Pausable
    struct PausableStorage {
        bool _paused;
    }

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.Pausable")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant PausableStorageLocation = 0xcd5ed15c6e187e77e9aee88184c21f4f2182ab5827cb3b7e07fbedcd63f03300;

    function _getPausableStorage() private pure returns (PausableStorage storage $) {
        assembly {
            $.slot := PausableStorageLocation
        }
    }

    /**
     * @dev Emitted when the pause is triggered by `account`.
     */
    event Paused(address account);

    /**
     * @dev Emitted when the pause is lifted by `account`.
     */
    event Unpaused(address account);

    /**
     * @dev The operation failed because the contract is paused.
     */
    error EnforcedPause();

    /**
     * @dev The operation failed because the contract is not paused.
     */
    error ExpectedPause();

    /**
     * @dev Initializes the contract in unpaused state.
     */
    function __Pausable_init() internal onlyInitializing {
        __Pausable_init_unchained();
    }

    function __Pausable_init_unchained() internal onlyInitializing {
        PausableStorage storage $ = _getPausableStorage();
        $._paused = false;
    }

    /**
     * @dev Modifier to make a function callable only when the contract is not paused.
     *
     * Requirements:
     *
     * - The contract must not be paused.
     */
    modifier whenNotPaused() {
        _requireNotPaused();
        _;
    }

    /**
     * @dev Modifier to make a function callable only when the contract is paused.
     *
     * Requirements:
     *
     * - The contract must be paused.
     */
    modifier whenPaused() {
        _requirePaused();
        _;
    }

    /**
     * @dev Returns true if the contract is paused, and false otherwise.
     */
    function paused() public view virtual returns (bool) {
        PausableStorage storage $ = _getPausableStorage();
        return $._paused;
    }

    /**
     * @dev Throws if the contract is paused.
     */
    function _requireNotPaused() internal view virtual {
        if (paused()) {
            revert EnforcedPause();
        }
    }

    /**
     * @dev Throws if the contract is not paused.
     */
    function _requirePaused() internal view virtual {
        if (!paused()) {
            revert ExpectedPause();
        }
    }

    /**
     * @dev Triggers stopped state.
     *
     * Requirements:
     *
     * - The contract must not be paused.
     */
    function _pause() internal virtual whenNotPaused {
        PausableStorage storage $ = _getPausableStorage();
        $._paused = true;
        emit Paused(_msgSender());
    }

    /**
     * @dev Returns to normal state.
     *
     * Requirements:
     *
     * - The contract must be paused.
     */
    function _unpause() internal virtual whenPaused {
        PausableStorage storage $ = _getPausableStorage();
        $._paused = false;
        emit Unpaused(_msgSender());
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts/utils/Address.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.2.0) (utils/Address.sol)

pragma solidity ^0.8.20;

import {Errors} from "./Errors.sol";

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

        (bool success, bytes memory returndata) = recipient.call{value: amount}("");
        if (!success) {
            _revert(returndata);
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
        (bool success, bytes memory returndata) = target.call{value: value}(data);
        return verifyCallResultFromTarget(target, success, returndata);
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but performing a static call.
     */
    function functionStaticCall(address target, bytes memory data) internal view returns (bytes memory) {
        (bool success, bytes memory returndata) = target.staticcall(data);
        return verifyCallResultFromTarget(target, success, returndata);
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but performing a delegate call.
     */
    function functionDelegateCall(address target, bytes memory data) internal returns (bytes memory) {
        (bool success, bytes memory returndata) = target.delegatecall(data);
        return verifyCallResultFromTarget(target, success, returndata);
    }

    /**
     * @dev Tool to verify that a low level call to smart-contract was successful, and reverts if the target
     * was not a contract or bubbling up the revert reason (falling back to {Errors.FailedCall}) in case
     * of an unsuccessful call.
     */
    function verifyCallResultFromTarget(
        address target,
        bool success,
        bytes memory returndata
    ) internal view returns (bytes memory) {
        if (!success) {
            _revert(returndata);
        } else {
            // only check if target is a contract if the call was successful and the return data is empty
            // otherwise we already know that it was a contract
            if (returndata.length == 0 && target.code.length == 0) {
                revert AddressEmptyCode(target);
            }
            return returndata;
        }
    }

    /**
     * @dev Tool to verify that a low level call was successful, and reverts if it wasn't, either by bubbling the
     * revert reason or with a default {Errors.FailedCall} error.
     */
    function verifyCallResult(bool success, bytes memory returndata) internal pure returns (bytes memory) {
        if (!success) {
            _revert(returndata);
        } else {
            return returndata;
        }
    }

    /**
     * @dev Reverts with returndata if present. Otherwise reverts with {Errors.FailedCall}.
     */
    function _revert(bytes memory returndata) private pure {
        // Look for revert reason and bubble it up if present
        if (returndata.length > 0) {
            // The easiest way to bubble the revert reason is using memory via assembly
            assembly ("memory-safe") {
                let returndata_size := mload(returndata)
                revert(add(32, returndata), returndata_size)
            }
        } else {
            revert Errors.FailedCall();
        }
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts/utils/structs/EnumerableSet.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/structs/EnumerableSet.sol)
// This file was procedurally generated from scripts/generate/templates/EnumerableSet.js.

pragma solidity ^0.8.20;

/**
 * @dev Library for managing
 * https://en.wikipedia.org/wiki/Set_(abstract_data_type)[sets] of primitive
 * types.
 *
 * Sets have the following properties:
 *
 * - Elements are added, removed, and checked for existence in constant time
 * (O(1)).
 * - Elements are enumerated in O(n). No guarantees are made on the ordering.
 *
 * ```solidity
 * contract Example {
 *     // Add the library methods
 *     using EnumerableSet for EnumerableSet.AddressSet;
 *
 *     // Declare a set state variable
 *     EnumerableSet.AddressSet private mySet;
 * }
 * ```
 *
 * As of v3.3.0, sets of type `bytes32` (`Bytes32Set`), `address` (`AddressSet`)
 * and `uint256` (`UintSet`) are supported.
 *
 * [WARNING]
 * ====
 * Trying to delete such a structure from storage will likely result in data corruption, rendering the structure
 * unusable.
 * See https://github.com/ethereum/solidity/pull/11843[ethereum/solidity#11843] for more info.
 *
 * In order to clean an EnumerableSet, you can either remove all elements one by one or create a fresh instance using an
 * array of EnumerableSet.
 * ====
 */
library EnumerableSet {
    // To implement this library for multiple types with as little code
    // repetition as possible, we write it in terms of a generic Set type with
    // bytes32 values.
    // The Set implementation uses private functions, and user-facing
    // implementations (such as AddressSet) are just wrappers around the
    // underlying Set.
    // This means that we can only create new EnumerableSets for types that fit
    // in bytes32.

    struct Set {
        // Storage of set values
        bytes32[] _values;
        // Position is the index of the value in the `values` array plus 1.
        // Position 0 is used to mean a value is not in the set.
        mapping(bytes32 value => uint256) _positions;
    }

    /**
     * @dev Add a value to a set. O(1).
     *
     * Returns true if the value was added to the set, that is if it was not
     * already present.
     */
    function _add(Set storage set, bytes32 value) private returns (bool) {
        if (!_contains(set, value)) {
            set._values.push(value);
            // The value is stored at length-1, but we add 1 to all indexes
            // and use 0 as a sentinel value
            set._positions[value] = set._values.length;
            return true;
        } else {
            return false;
        }
    }

    /**
     * @dev Removes a value from a set. O(1).
     *
     * Returns true if the value was removed from the set, that is if it was
     * present.
     */
    function _remove(Set storage set, bytes32 value) private returns (bool) {
        // We cache the value's position to prevent multiple reads from the same storage slot
        uint256 position = set._positions[value];

        if (position != 0) {
            // Equivalent to contains(set, value)
            // To delete an element from the _values array in O(1), we swap the element to delete with the last one in
            // the array, and then remove the last element (sometimes called as 'swap and pop').
            // This modifies the order of the array, as noted in {at}.

            uint256 valueIndex = position - 1;
            uint256 lastIndex = set._values.length - 1;

            if (valueIndex != lastIndex) {
                bytes32 lastValue = set._values[lastIndex];

                // Move the lastValue to the index where the value to delete is
                set._values[valueIndex] = lastValue;
                // Update the tracked position of the lastValue (that was just moved)
                set._positions[lastValue] = position;
            }

            // Delete the slot where the moved value was stored
            set._values.pop();

            // Delete the tracked position for the deleted slot
            delete set._positions[value];

            return true;
        } else {
            return false;
        }
    }

    /**
     * @dev Returns true if the value is in the set. O(1).
     */
    function _contains(Set storage set, bytes32 value) private view returns (bool) {
        return set._positions[value] != 0;
    }

    /**
     * @dev Returns the number of values on the set. O(1).
     */
    function _length(Set storage set) private view returns (uint256) {
        return set._values.length;
    }

    /**
     * @dev Returns the value stored at position `index` in the set. O(1).
     *
     * Note that there are no guarantees on the ordering of values inside the
     * array, and it may change when more values are added or removed.
     *
     * Requirements:
     *
     * - `index` must be strictly less than {length}.
     */
    function _at(Set storage set, uint256 index) private view returns (bytes32) {
        return set._values[index];
    }

    /**
     * @dev Return the entire set in an array
     *
     * WARNING: This operation will copy the entire storage to memory, which can be quite expensive. This is designed
     * to mostly be used by view accessors that are queried without any gas fees. Developers should keep in mind that
     * this function has an unbounded cost, and using it as part of a state-changing function may render the function
     * uncallable if the set grows to a point where copying to memory consumes too much gas to fit in a block.
     */
    function _values(Set storage set) private view returns (bytes32[] memory) {
        return set._values;
    }

    // Bytes32Set

    struct Bytes32Set {
        Set _inner;
    }

    /**
     * @dev Add a value to a set. O(1).
     *
     * Returns true if the value was added to the set, that is if it was not
     * already present.
     */
    function add(Bytes32Set storage set, bytes32 value) internal returns (bool) {
        return _add(set._inner, value);
    }

    /**
     * @dev Removes a value from a set. O(1).
     *
     * Returns true if the value was removed from the set, that is if it was
     * present.
     */
    function remove(Bytes32Set storage set, bytes32 value) internal returns (bool) {
        return _remove(set._inner, value);
    }

    /**
     * @dev Returns true if the value is in the set. O(1).
     */
    function contains(Bytes32Set storage set, bytes32 value) internal view returns (bool) {
        return _contains(set._inner, value);
    }

    /**
     * @dev Returns the number of values in the set. O(1).
     */
    function length(Bytes32Set storage set) internal view returns (uint256) {
        return _length(set._inner);
    }

    /**
     * @dev Returns the value stored at position `index` in the set. O(1).
     *
     * Note that there are no guarantees on the ordering of values inside the
     * array, and it may change when more values are added or removed.
     *
     * Requirements:
     *
     * - `index` must be strictly less than {length}.
     */
    function at(Bytes32Set storage set, uint256 index) internal view returns (bytes32) {
        return _at(set._inner, index);
    }

    /**
     * @dev Return the entire set in an array
     *
     * WARNING: This operation will copy the entire storage to memory, which can be quite expensive. This is designed
     * to mostly be used by view accessors that are queried without any gas fees. Developers should keep in mind that
     * this function has an unbounded cost, and using it as part of a state-changing function may render the function
     * uncallable if the set grows to a point where copying to memory consumes too much gas to fit in a block.
     */
    function values(Bytes32Set storage set) internal view returns (bytes32[] memory) {
        bytes32[] memory store = _values(set._inner);
        bytes32[] memory result;

        assembly ("memory-safe") {
            result := store
        }

        return result;
    }

    // AddressSet

    struct AddressSet {
        Set _inner;
    }

    /**
     * @dev Add a value to a set. O(1).
     *
     * Returns true if the value was added to the set, that is if it was not
     * already present.
     */
    function add(AddressSet storage set, address value) internal returns (bool) {
        return _add(set._inner, bytes32(uint256(uint160(value))));
    }

    /**
     * @dev Removes a value from a set. O(1).
     *
     * Returns true if the value was removed from the set, that is if it was
     * present.
     */
    function remove(AddressSet storage set, address value) internal returns (bool) {
        return _remove(set._inner, bytes32(uint256(uint160(value))));
    }

    /**
     * @dev Returns true if the value is in the set. O(1).
     */
    function contains(AddressSet storage set, address value) internal view returns (bool) {
        return _contains(set._inner, bytes32(uint256(uint160(value))));
    }

    /**
     * @dev Returns the number of values in the set. O(1).
     */
    function length(AddressSet storage set) internal view returns (uint256) {
        return _length(set._inner);
    }

    /**
     * @dev Returns the value stored at position `index` in the set. O(1).
     *
     * Note that there are no guarantees on the ordering of values inside the
     * array, and it may change when more values are added or removed.
     *
     * Requirements:
     *
     * - `index` must be strictly less than {length}.
     */
    function at(AddressSet storage set, uint256 index) internal view returns (address) {
        return address(uint160(uint256(_at(set._inner, index))));
    }

    /**
     * @dev Return the entire set in an array
     *
     * WARNING: This operation will copy the entire storage to memory, which can be quite expensive. This is designed
     * to mostly be used by view accessors that are queried without any gas fees. Developers should keep in mind that
     * this function has an unbounded cost, and using it as part of a state-changing function may render the function
     * uncallable if the set grows to a point where copying to memory consumes too much gas to fit in a block.
     */
    function values(AddressSet storage set) internal view returns (address[] memory) {
        bytes32[] memory store = _values(set._inner);
        address[] memory result;

        assembly ("memory-safe") {
            result := store
        }

        return result;
    }

    // UintSet

    struct UintSet {
        Set _inner;
    }

    /**
     * @dev Add a value to a set. O(1).
     *
     * Returns true if the value was added to the set, that is if it was not
     * already present.
     */
    function add(UintSet storage set, uint256 value) internal returns (bool) {
        return _add(set._inner, bytes32(value));
    }

    /**
     * @dev Removes a value from a set. O(1).
     *
     * Returns true if the value was removed from the set, that is if it was
     * present.
     */
    function remove(UintSet storage set, uint256 value) internal returns (bool) {
        return _remove(set._inner, bytes32(value));
    }

    /**
     * @dev Returns true if the value is in the set. O(1).
     */
    function contains(UintSet storage set, uint256 value) internal view returns (bool) {
        return _contains(set._inner, bytes32(value));
    }

    /**
     * @dev Returns the number of values in the set. O(1).
     */
    function length(UintSet storage set) internal view returns (uint256) {
        return _length(set._inner);
    }

    /**
     * @dev Returns the value stored at position `index` in the set. O(1).
     *
     * Note that there are no guarantees on the ordering of values inside the
     * array, and it may change when more values are added or removed.
     *
     * Requirements:
     *
     * - `index` must be strictly less than {length}.
     */
    function at(UintSet storage set, uint256 index) internal view returns (uint256) {
        return uint256(_at(set._inner, index));
    }

    /**
     * @dev Return the entire set in an array
     *
     * WARNING: This operation will copy the entire storage to memory, which can be quite expensive. This is designed
     * to mostly be used by view accessors that are queried without any gas fees. Developers should keep in mind that
     * this function has an unbounded cost, and using it as part of a state-changing function may render the function
     * uncallable if the set grows to a point where copying to memory consumes too much gas to fit in a block.
     */
    function values(UintSet storage set) internal view returns (uint256[] memory) {
        bytes32[] memory store = _values(set._inner);
        uint256[] memory result;

        assembly ("memory-safe") {
            result := store
        }

        return result;
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts/access/Ownable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (access/Ownable.sol)

pragma solidity ^0.8.20;

import {Context} from "../utils/Context.sol";

/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * The initial owner is set to the address provided by the deployer. This can
 * later be changed with {transferOwnership}.
 *
 * This module is used through inheritance. It will make available the modifier
 * `onlyOwner`, which can be applied to your functions to restrict their use to
 * the owner.
 */
abstract contract Ownable is Context {
    address private _owner;

    /**
     * @dev The caller account is not authorized to perform an operation.
     */
    error OwnableUnauthorizedAccount(address account);

    /**
     * @dev The owner is not a valid owner account. (eg. `address(0)`)
     */
    error OwnableInvalidOwner(address owner);

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev Initializes the contract setting the address provided by the deployer as the initial owner.
     */
    constructor(address initialOwner) {
        if (initialOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(initialOwner);
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    /**
     * @dev Returns the address of the current owner.
     */
    function owner() public view virtual returns (address) {
        return _owner;
    }

    /**
     * @dev Throws if the sender is not the owner.
     */
    function _checkOwner() internal view virtual {
        if (owner() != _msgSender()) {
            revert OwnableUnauthorizedAccount(_msgSender());
        }
    }

    /**
     * @dev Leaves the contract without owner. It will not be possible to call
     * `onlyOwner` functions. Can only be called by the current owner.
     *
     * NOTE: Renouncing ownership will leave the contract without an owner,
     * thereby disabling any functionality that is only available to the owner.
     */
    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(newOwner);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Internal function without access restriction.
     */
    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts/utils/math/SafeCast.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/math/SafeCast.sol)
// This file was procedurally generated from scripts/generate/templates/SafeCast.js.

pragma solidity ^0.8.20;

/**
 * @dev Wrappers over Solidity's uintXX/intXX/bool casting operators with added overflow
 * checks.
 *
 * Downcasting from uint256/int256 in Solidity does not revert on overflow. This can
 * easily result in undesired exploitation or bugs, since developers usually
 * assume that overflows raise errors. `SafeCast` restores this intuition by
 * reverting the transaction when such an operation overflows.
 *
 * Using this library instead of the unchecked operations eliminates an entire
 * class of bugs, so it's recommended to use it always.
 */
library SafeCast {
    /**
     * @dev Value doesn't fit in an uint of `bits` size.
     */
    error SafeCastOverflowedUintDowncast(uint8 bits, uint256 value);

    /**
     * @dev An int value doesn't fit in an uint of `bits` size.
     */
    error SafeCastOverflowedIntToUint(int256 value);

    /**
     * @dev Value doesn't fit in an int of `bits` size.
     */
    error SafeCastOverflowedIntDowncast(uint8 bits, int256 value);

    /**
     * @dev An uint value doesn't fit in an int of `bits` size.
     */
    error SafeCastOverflowedUintToInt(uint256 value);

    /**
     * @dev Returns the downcasted uint248 from uint256, reverting on
     * overflow (when the input is greater than largest uint248).
     *
     * Counterpart to Solidity's `uint248` operator.
     *
     * Requirements:
     *
     * - input must fit into 248 bits
     */
    function toUint248(uint256 value) internal pure returns (uint248) {
        if (value > type(uint248).max) {
            revert SafeCastOverflowedUintDowncast(248, value);
        }
        return uint248(value);
    }

    /**
     * @dev Returns the downcasted uint240 from uint256, reverting on
     * overflow (when the input is greater than largest uint240).
     *
     * Counterpart to Solidity's `uint240` operator.
     *
     * Requirements:
     *
     * - input must fit into 240 bits
     */
    function toUint240(uint256 value) internal pure returns (uint240) {
        if (value > type(uint240).max) {
            revert SafeCastOverflowedUintDowncast(240, value);
        }
        return uint240(value);
    }

    /**
     * @dev Returns the downcasted uint232 from uint256, reverting on
     * overflow (when the input is greater than largest uint232).
     *
     * Counterpart to Solidity's `uint232` operator.
     *
     * Requirements:
     *
     * - input must fit into 232 bits
     */
    function toUint232(uint256 value) internal pure returns (uint232) {
        if (value > type(uint232).max) {
            revert SafeCastOverflowedUintDowncast(232, value);
        }
        return uint232(value);
    }

    /**
     * @dev Returns the downcasted uint224 from uint256, reverting on
     * overflow (when the input is greater than largest uint224).
     *
     * Counterpart to Solidity's `uint224` operator.
     *
     * Requirements:
     *
     * - input must fit into 224 bits
     */
    function toUint224(uint256 value) internal pure returns (uint224) {
        if (value > type(uint224).max) {
            revert SafeCastOverflowedUintDowncast(224, value);
        }
        return uint224(value);
    }

    /**
     * @dev Returns the downcasted uint216 from uint256, reverting on
     * overflow (when the input is greater than largest uint216).
     *
     * Counterpart to Solidity's `uint216` operator.
     *
     * Requirements:
     *
     * - input must fit into 216 bits
     */
    function toUint216(uint256 value) internal pure returns (uint216) {
        if (value > type(uint216).max) {
            revert SafeCastOverflowedUintDowncast(216, value);
        }
        return uint216(value);
    }

    /**
     * @dev Returns the downcasted uint208 from uint256, reverting on
     * overflow (when the input is greater than largest uint208).
     *
     * Counterpart to Solidity's `uint208` operator.
     *
     * Requirements:
     *
     * - input must fit into 208 bits
     */
    function toUint208(uint256 value) internal pure returns (uint208) {
        if (value > type(uint208).max) {
            revert SafeCastOverflowedUintDowncast(208, value);
        }
        return uint208(value);
    }

    /**
     * @dev Returns the downcasted uint200 from uint256, reverting on
     * overflow (when the input is greater than largest uint200).
     *
     * Counterpart to Solidity's `uint200` operator.
     *
     * Requirements:
     *
     * - input must fit into 200 bits
     */
    function toUint200(uint256 value) internal pure returns (uint200) {
        if (value > type(uint200).max) {
            revert SafeCastOverflowedUintDowncast(200, value);
        }
        return uint200(value);
    }

    /**
     * @dev Returns the downcasted uint192 from uint256, reverting on
     * overflow (when the input is greater than largest uint192).
     *
     * Counterpart to Solidity's `uint192` operator.
     *
     * Requirements:
     *
     * - input must fit into 192 bits
     */
    function toUint192(uint256 value) internal pure returns (uint192) {
        if (value > type(uint192).max) {
            revert SafeCastOverflowedUintDowncast(192, value);
        }
        return uint192(value);
    }

    /**
     * @dev Returns the downcasted uint184 from uint256, reverting on
     * overflow (when the input is greater than largest uint184).
     *
     * Counterpart to Solidity's `uint184` operator.
     *
     * Requirements:
     *
     * - input must fit into 184 bits
     */
    function toUint184(uint256 value) internal pure returns (uint184) {
        if (value > type(uint184).max) {
            revert SafeCastOverflowedUintDowncast(184, value);
        }
        return uint184(value);
    }

    /**
     * @dev Returns the downcasted uint176 from uint256, reverting on
     * overflow (when the input is greater than largest uint176).
     *
     * Counterpart to Solidity's `uint176` operator.
     *
     * Requirements:
     *
     * - input must fit into 176 bits
     */
    function toUint176(uint256 value) internal pure returns (uint176) {
        if (value > type(uint176).max) {
            revert SafeCastOverflowedUintDowncast(176, value);
        }
        return uint176(value);
    }

    /**
     * @dev Returns the downcasted uint168 from uint256, reverting on
     * overflow (when the input is greater than largest uint168).
     *
     * Counterpart to Solidity's `uint168` operator.
     *
     * Requirements:
     *
     * - input must fit into 168 bits
     */
    function toUint168(uint256 value) internal pure returns (uint168) {
        if (value > type(uint168).max) {
            revert SafeCastOverflowedUintDowncast(168, value);
        }
        return uint168(value);
    }

    /**
     * @dev Returns the downcasted uint160 from uint256, reverting on
     * overflow (when the input is greater than largest uint160).
     *
     * Counterpart to Solidity's `uint160` operator.
     *
     * Requirements:
     *
     * - input must fit into 160 bits
     */
    function toUint160(uint256 value) internal pure returns (uint160) {
        if (value > type(uint160).max) {
            revert SafeCastOverflowedUintDowncast(160, value);
        }
        return uint160(value);
    }

    /**
     * @dev Returns the downcasted uint152 from uint256, reverting on
     * overflow (when the input is greater than largest uint152).
     *
     * Counterpart to Solidity's `uint152` operator.
     *
     * Requirements:
     *
     * - input must fit into 152 bits
     */
    function toUint152(uint256 value) internal pure returns (uint152) {
        if (value > type(uint152).max) {
            revert SafeCastOverflowedUintDowncast(152, value);
        }
        return uint152(value);
    }

    /**
     * @dev Returns the downcasted uint144 from uint256, reverting on
     * overflow (when the input is greater than largest uint144).
     *
     * Counterpart to Solidity's `uint144` operator.
     *
     * Requirements:
     *
     * - input must fit into 144 bits
     */
    function toUint144(uint256 value) internal pure returns (uint144) {
        if (value > type(uint144).max) {
            revert SafeCastOverflowedUintDowncast(144, value);
        }
        return uint144(value);
    }

    /**
     * @dev Returns the downcasted uint136 from uint256, reverting on
     * overflow (when the input is greater than largest uint136).
     *
     * Counterpart to Solidity's `uint136` operator.
     *
     * Requirements:
     *
     * - input must fit into 136 bits
     */
    function toUint136(uint256 value) internal pure returns (uint136) {
        if (value > type(uint136).max) {
            revert SafeCastOverflowedUintDowncast(136, value);
        }
        return uint136(value);
    }

    /**
     * @dev Returns the downcasted uint128 from uint256, reverting on
     * overflow (when the input is greater than largest uint128).
     *
     * Counterpart to Solidity's `uint128` operator.
     *
     * Requirements:
     *
     * - input must fit into 128 bits
     */
    function toUint128(uint256 value) internal pure returns (uint128) {
        if (value > type(uint128).max) {
            revert SafeCastOverflowedUintDowncast(128, value);
        }
        return uint128(value);
    }

    /**
     * @dev Returns the downcasted uint120 from uint256, reverting on
     * overflow (when the input is greater than largest uint120).
     *
     * Counterpart to Solidity's `uint120` operator.
     *
     * Requirements:
     *
     * - input must fit into 120 bits
     */
    function toUint120(uint256 value) internal pure returns (uint120) {
        if (value > type(uint120).max) {
            revert SafeCastOverflowedUintDowncast(120, value);
        }
        return uint120(value);
    }

    /**
     * @dev Returns the downcasted uint112 from uint256, reverting on
     * overflow (when the input is greater than largest uint112).
     *
     * Counterpart to Solidity's `uint112` operator.
     *
     * Requirements:
     *
     * - input must fit into 112 bits
     */
    function toUint112(uint256 value) internal pure returns (uint112) {
        if (value > type(uint112).max) {
            revert SafeCastOverflowedUintDowncast(112, value);
        }
        return uint112(value);
    }

    /**
     * @dev Returns the downcasted uint104 from uint256, reverting on
     * overflow (when the input is greater than largest uint104).
     *
     * Counterpart to Solidity's `uint104` operator.
     *
     * Requirements:
     *
     * - input must fit into 104 bits
     */
    function toUint104(uint256 value) internal pure returns (uint104) {
        if (value > type(uint104).max) {
            revert SafeCastOverflowedUintDowncast(104, value);
        }
        return uint104(value);
    }

    /**
     * @dev Returns the downcasted uint96 from uint256, reverting on
     * overflow (when the input is greater than largest uint96).
     *
     * Counterpart to Solidity's `uint96` operator.
     *
     * Requirements:
     *
     * - input must fit into 96 bits
     */
    function toUint96(uint256 value) internal pure returns (uint96) {
        if (value > type(uint96).max) {
            revert SafeCastOverflowedUintDowncast(96, value);
        }
        return uint96(value);
    }

    /**
     * @dev Returns the downcasted uint88 from uint256, reverting on
     * overflow (when the input is greater than largest uint88).
     *
     * Counterpart to Solidity's `uint88` operator.
     *
     * Requirements:
     *
     * - input must fit into 88 bits
     */
    function toUint88(uint256 value) internal pure returns (uint88) {
        if (value > type(uint88).max) {
            revert SafeCastOverflowedUintDowncast(88, value);
        }
        return uint88(value);
    }

    /**
     * @dev Returns the downcasted uint80 from uint256, reverting on
     * overflow (when the input is greater than largest uint80).
     *
     * Counterpart to Solidity's `uint80` operator.
     *
     * Requirements:
     *
     * - input must fit into 80 bits
     */
    function toUint80(uint256 value) internal pure returns (uint80) {
        if (value > type(uint80).max) {
            revert SafeCastOverflowedUintDowncast(80, value);
        }
        return uint80(value);
    }

    /**
     * @dev Returns the downcasted uint72 from uint256, reverting on
     * overflow (when the input is greater than largest uint72).
     *
     * Counterpart to Solidity's `uint72` operator.
     *
     * Requirements:
     *
     * - input must fit into 72 bits
     */
    function toUint72(uint256 value) internal pure returns (uint72) {
        if (value > type(uint72).max) {
            revert SafeCastOverflowedUintDowncast(72, value);
        }
        return uint72(value);
    }

    /**
     * @dev Returns the downcasted uint64 from uint256, reverting on
     * overflow (when the input is greater than largest uint64).
     *
     * Counterpart to Solidity's `uint64` operator.
     *
     * Requirements:
     *
     * - input must fit into 64 bits
     */
    function toUint64(uint256 value) internal pure returns (uint64) {
        if (value > type(uint64).max) {
            revert SafeCastOverflowedUintDowncast(64, value);
        }
        return uint64(value);
    }

    /**
     * @dev Returns the downcasted uint56 from uint256, reverting on
     * overflow (when the input is greater than largest uint56).
     *
     * Counterpart to Solidity's `uint56` operator.
     *
     * Requirements:
     *
     * - input must fit into 56 bits
     */
    function toUint56(uint256 value) internal pure returns (uint56) {
        if (value > type(uint56).max) {
            revert SafeCastOverflowedUintDowncast(56, value);
        }
        return uint56(value);
    }

    /**
     * @dev Returns the downcasted uint48 from uint256, reverting on
     * overflow (when the input is greater than largest uint48).
     *
     * Counterpart to Solidity's `uint48` operator.
     *
     * Requirements:
     *
     * - input must fit into 48 bits
     */
    function toUint48(uint256 value) internal pure returns (uint48) {
        if (value > type(uint48).max) {
            revert SafeCastOverflowedUintDowncast(48, value);
        }
        return uint48(value);
    }

    /**
     * @dev Returns the downcasted uint40 from uint256, reverting on
     * overflow (when the input is greater than largest uint40).
     *
     * Counterpart to Solidity's `uint40` operator.
     *
     * Requirements:
     *
     * - input must fit into 40 bits
     */
    function toUint40(uint256 value) internal pure returns (uint40) {
        if (value > type(uint40).max) {
            revert SafeCastOverflowedUintDowncast(40, value);
        }
        return uint40(value);
    }

    /**
     * @dev Returns the downcasted uint32 from uint256, reverting on
     * overflow (when the input is greater than largest uint32).
     *
     * Counterpart to Solidity's `uint32` operator.
     *
     * Requirements:
     *
     * - input must fit into 32 bits
     */
    function toUint32(uint256 value) internal pure returns (uint32) {
        if (value > type(uint32).max) {
            revert SafeCastOverflowedUintDowncast(32, value);
        }
        return uint32(value);
    }

    /**
     * @dev Returns the downcasted uint24 from uint256, reverting on
     * overflow (when the input is greater than largest uint24).
     *
     * Counterpart to Solidity's `uint24` operator.
     *
     * Requirements:
     *
     * - input must fit into 24 bits
     */
    function toUint24(uint256 value) internal pure returns (uint24) {
        if (value > type(uint24).max) {
            revert SafeCastOverflowedUintDowncast(24, value);
        }
        return uint24(value);
    }

    /**
     * @dev Returns the downcasted uint16 from uint256, reverting on
     * overflow (when the input is greater than largest uint16).
     *
     * Counterpart to Solidity's `uint16` operator.
     *
     * Requirements:
     *
     * - input must fit into 16 bits
     */
    function toUint16(uint256 value) internal pure returns (uint16) {
        if (value > type(uint16).max) {
            revert SafeCastOverflowedUintDowncast(16, value);
        }
        return uint16(value);
    }

    /**
     * @dev Returns the downcasted uint8 from uint256, reverting on
     * overflow (when the input is greater than largest uint8).
     *
     * Counterpart to Solidity's `uint8` operator.
     *
     * Requirements:
     *
     * - input must fit into 8 bits
     */
    function toUint8(uint256 value) internal pure returns (uint8) {
        if (value > type(uint8).max) {
            revert SafeCastOverflowedUintDowncast(8, value);
        }
        return uint8(value);
    }

    /**
     * @dev Converts a signed int256 into an unsigned uint256.
     *
     * Requirements:
     *
     * - input must be greater than or equal to 0.
     */
    function toUint256(int256 value) internal pure returns (uint256) {
        if (value < 0) {
            revert SafeCastOverflowedIntToUint(value);
        }
        return uint256(value);
    }

    /**
     * @dev Returns the downcasted int248 from int256, reverting on
     * overflow (when the input is less than smallest int248 or
     * greater than largest int248).
     *
     * Counterpart to Solidity's `int248` operator.
     *
     * Requirements:
     *
     * - input must fit into 248 bits
     */
    function toInt248(int256 value) internal pure returns (int248 downcasted) {
        downcasted = int248(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(248, value);
        }
    }

    /**
     * @dev Returns the downcasted int240 from int256, reverting on
     * overflow (when the input is less than smallest int240 or
     * greater than largest int240).
     *
     * Counterpart to Solidity's `int240` operator.
     *
     * Requirements:
     *
     * - input must fit into 240 bits
     */
    function toInt240(int256 value) internal pure returns (int240 downcasted) {
        downcasted = int240(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(240, value);
        }
    }

    /**
     * @dev Returns the downcasted int232 from int256, reverting on
     * overflow (when the input is less than smallest int232 or
     * greater than largest int232).
     *
     * Counterpart to Solidity's `int232` operator.
     *
     * Requirements:
     *
     * - input must fit into 232 bits
     */
    function toInt232(int256 value) internal pure returns (int232 downcasted) {
        downcasted = int232(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(232, value);
        }
    }

    /**
     * @dev Returns the downcasted int224 from int256, reverting on
     * overflow (when the input is less than smallest int224 or
     * greater than largest int224).
     *
     * Counterpart to Solidity's `int224` operator.
     *
     * Requirements:
     *
     * - input must fit into 224 bits
     */
    function toInt224(int256 value) internal pure returns (int224 downcasted) {
        downcasted = int224(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(224, value);
        }
    }

    /**
     * @dev Returns the downcasted int216 from int256, reverting on
     * overflow (when the input is less than smallest int216 or
     * greater than largest int216).
     *
     * Counterpart to Solidity's `int216` operator.
     *
     * Requirements:
     *
     * - input must fit into 216 bits
     */
    function toInt216(int256 value) internal pure returns (int216 downcasted) {
        downcasted = int216(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(216, value);
        }
    }

    /**
     * @dev Returns the downcasted int208 from int256, reverting on
     * overflow (when the input is less than smallest int208 or
     * greater than largest int208).
     *
     * Counterpart to Solidity's `int208` operator.
     *
     * Requirements:
     *
     * - input must fit into 208 bits
     */
    function toInt208(int256 value) internal pure returns (int208 downcasted) {
        downcasted = int208(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(208, value);
        }
    }

    /**
     * @dev Returns the downcasted int200 from int256, reverting on
     * overflow (when the input is less than smallest int200 or
     * greater than largest int200).
     *
     * Counterpart to Solidity's `int200` operator.
     *
     * Requirements:
     *
     * - input must fit into 200 bits
     */
    function toInt200(int256 value) internal pure returns (int200 downcasted) {
        downcasted = int200(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(200, value);
        }
    }

    /**
     * @dev Returns the downcasted int192 from int256, reverting on
     * overflow (when the input is less than smallest int192 or
     * greater than largest int192).
     *
     * Counterpart to Solidity's `int192` operator.
     *
     * Requirements:
     *
     * - input must fit into 192 bits
     */
    function toInt192(int256 value) internal pure returns (int192 downcasted) {
        downcasted = int192(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(192, value);
        }
    }

    /**
     * @dev Returns the downcasted int184 from int256, reverting on
     * overflow (when the input is less than smallest int184 or
     * greater than largest int184).
     *
     * Counterpart to Solidity's `int184` operator.
     *
     * Requirements:
     *
     * - input must fit into 184 bits
     */
    function toInt184(int256 value) internal pure returns (int184 downcasted) {
        downcasted = int184(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(184, value);
        }
    }

    /**
     * @dev Returns the downcasted int176 from int256, reverting on
     * overflow (when the input is less than smallest int176 or
     * greater than largest int176).
     *
     * Counterpart to Solidity's `int176` operator.
     *
     * Requirements:
     *
     * - input must fit into 176 bits
     */
    function toInt176(int256 value) internal pure returns (int176 downcasted) {
        downcasted = int176(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(176, value);
        }
    }

    /**
     * @dev Returns the downcasted int168 from int256, reverting on
     * overflow (when the input is less than smallest int168 or
     * greater than largest int168).
     *
     * Counterpart to Solidity's `int168` operator.
     *
     * Requirements:
     *
     * - input must fit into 168 bits
     */
    function toInt168(int256 value) internal pure returns (int168 downcasted) {
        downcasted = int168(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(168, value);
        }
    }

    /**
     * @dev Returns the downcasted int160 from int256, reverting on
     * overflow (when the input is less than smallest int160 or
     * greater than largest int160).
     *
     * Counterpart to Solidity's `int160` operator.
     *
     * Requirements:
     *
     * - input must fit into 160 bits
     */
    function toInt160(int256 value) internal pure returns (int160 downcasted) {
        downcasted = int160(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(160, value);
        }
    }

    /**
     * @dev Returns the downcasted int152 from int256, reverting on
     * overflow (when the input is less than smallest int152 or
     * greater than largest int152).
     *
     * Counterpart to Solidity's `int152` operator.
     *
     * Requirements:
     *
     * - input must fit into 152 bits
     */
    function toInt152(int256 value) internal pure returns (int152 downcasted) {
        downcasted = int152(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(152, value);
        }
    }

    /**
     * @dev Returns the downcasted int144 from int256, reverting on
     * overflow (when the input is less than smallest int144 or
     * greater than largest int144).
     *
     * Counterpart to Solidity's `int144` operator.
     *
     * Requirements:
     *
     * - input must fit into 144 bits
     */
    function toInt144(int256 value) internal pure returns (int144 downcasted) {
        downcasted = int144(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(144, value);
        }
    }

    /**
     * @dev Returns the downcasted int136 from int256, reverting on
     * overflow (when the input is less than smallest int136 or
     * greater than largest int136).
     *
     * Counterpart to Solidity's `int136` operator.
     *
     * Requirements:
     *
     * - input must fit into 136 bits
     */
    function toInt136(int256 value) internal pure returns (int136 downcasted) {
        downcasted = int136(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(136, value);
        }
    }

    /**
     * @dev Returns the downcasted int128 from int256, reverting on
     * overflow (when the input is less than smallest int128 or
     * greater than largest int128).
     *
     * Counterpart to Solidity's `int128` operator.
     *
     * Requirements:
     *
     * - input must fit into 128 bits
     */
    function toInt128(int256 value) internal pure returns (int128 downcasted) {
        downcasted = int128(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(128, value);
        }
    }

    /**
     * @dev Returns the downcasted int120 from int256, reverting on
     * overflow (when the input is less than smallest int120 or
     * greater than largest int120).
     *
     * Counterpart to Solidity's `int120` operator.
     *
     * Requirements:
     *
     * - input must fit into 120 bits
     */
    function toInt120(int256 value) internal pure returns (int120 downcasted) {
        downcasted = int120(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(120, value);
        }
    }

    /**
     * @dev Returns the downcasted int112 from int256, reverting on
     * overflow (when the input is less than smallest int112 or
     * greater than largest int112).
     *
     * Counterpart to Solidity's `int112` operator.
     *
     * Requirements:
     *
     * - input must fit into 112 bits
     */
    function toInt112(int256 value) internal pure returns (int112 downcasted) {
        downcasted = int112(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(112, value);
        }
    }

    /**
     * @dev Returns the downcasted int104 from int256, reverting on
     * overflow (when the input is less than smallest int104 or
     * greater than largest int104).
     *
     * Counterpart to Solidity's `int104` operator.
     *
     * Requirements:
     *
     * - input must fit into 104 bits
     */
    function toInt104(int256 value) internal pure returns (int104 downcasted) {
        downcasted = int104(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(104, value);
        }
    }

    /**
     * @dev Returns the downcasted int96 from int256, reverting on
     * overflow (when the input is less than smallest int96 or
     * greater than largest int96).
     *
     * Counterpart to Solidity's `int96` operator.
     *
     * Requirements:
     *
     * - input must fit into 96 bits
     */
    function toInt96(int256 value) internal pure returns (int96 downcasted) {
        downcasted = int96(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(96, value);
        }
    }

    /**
     * @dev Returns the downcasted int88 from int256, reverting on
     * overflow (when the input is less than smallest int88 or
     * greater than largest int88).
     *
     * Counterpart to Solidity's `int88` operator.
     *
     * Requirements:
     *
     * - input must fit into 88 bits
     */
    function toInt88(int256 value) internal pure returns (int88 downcasted) {
        downcasted = int88(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(88, value);
        }
    }

    /**
     * @dev Returns the downcasted int80 from int256, reverting on
     * overflow (when the input is less than smallest int80 or
     * greater than largest int80).
     *
     * Counterpart to Solidity's `int80` operator.
     *
     * Requirements:
     *
     * - input must fit into 80 bits
     */
    function toInt80(int256 value) internal pure returns (int80 downcasted) {
        downcasted = int80(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(80, value);
        }
    }

    /**
     * @dev Returns the downcasted int72 from int256, reverting on
     * overflow (when the input is less than smallest int72 or
     * greater than largest int72).
     *
     * Counterpart to Solidity's `int72` operator.
     *
     * Requirements:
     *
     * - input must fit into 72 bits
     */
    function toInt72(int256 value) internal pure returns (int72 downcasted) {
        downcasted = int72(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(72, value);
        }
    }

    /**
     * @dev Returns the downcasted int64 from int256, reverting on
     * overflow (when the input is less than smallest int64 or
     * greater than largest int64).
     *
     * Counterpart to Solidity's `int64` operator.
     *
     * Requirements:
     *
     * - input must fit into 64 bits
     */
    function toInt64(int256 value) internal pure returns (int64 downcasted) {
        downcasted = int64(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(64, value);
        }
    }

    /**
     * @dev Returns the downcasted int56 from int256, reverting on
     * overflow (when the input is less than smallest int56 or
     * greater than largest int56).
     *
     * Counterpart to Solidity's `int56` operator.
     *
     * Requirements:
     *
     * - input must fit into 56 bits
     */
    function toInt56(int256 value) internal pure returns (int56 downcasted) {
        downcasted = int56(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(56, value);
        }
    }

    /**
     * @dev Returns the downcasted int48 from int256, reverting on
     * overflow (when the input is less than smallest int48 or
     * greater than largest int48).
     *
     * Counterpart to Solidity's `int48` operator.
     *
     * Requirements:
     *
     * - input must fit into 48 bits
     */
    function toInt48(int256 value) internal pure returns (int48 downcasted) {
        downcasted = int48(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(48, value);
        }
    }

    /**
     * @dev Returns the downcasted int40 from int256, reverting on
     * overflow (when the input is less than smallest int40 or
     * greater than largest int40).
     *
     * Counterpart to Solidity's `int40` operator.
     *
     * Requirements:
     *
     * - input must fit into 40 bits
     */
    function toInt40(int256 value) internal pure returns (int40 downcasted) {
        downcasted = int40(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(40, value);
        }
    }

    /**
     * @dev Returns the downcasted int32 from int256, reverting on
     * overflow (when the input is less than smallest int32 or
     * greater than largest int32).
     *
     * Counterpart to Solidity's `int32` operator.
     *
     * Requirements:
     *
     * - input must fit into 32 bits
     */
    function toInt32(int256 value) internal pure returns (int32 downcasted) {
        downcasted = int32(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(32, value);
        }
    }

    /**
     * @dev Returns the downcasted int24 from int256, reverting on
     * overflow (when the input is less than smallest int24 or
     * greater than largest int24).
     *
     * Counterpart to Solidity's `int24` operator.
     *
     * Requirements:
     *
     * - input must fit into 24 bits
     */
    function toInt24(int256 value) internal pure returns (int24 downcasted) {
        downcasted = int24(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(24, value);
        }
    }

    /**
     * @dev Returns the downcasted int16 from int256, reverting on
     * overflow (when the input is less than smallest int16 or
     * greater than largest int16).
     *
     * Counterpart to Solidity's `int16` operator.
     *
     * Requirements:
     *
     * - input must fit into 16 bits
     */
    function toInt16(int256 value) internal pure returns (int16 downcasted) {
        downcasted = int16(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(16, value);
        }
    }

    /**
     * @dev Returns the downcasted int8 from int256, reverting on
     * overflow (when the input is less than smallest int8 or
     * greater than largest int8).
     *
     * Counterpart to Solidity's `int8` operator.
     *
     * Requirements:
     *
     * - input must fit into 8 bits
     */
    function toInt8(int256 value) internal pure returns (int8 downcasted) {
        downcasted = int8(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(8, value);
        }
    }

    /**
     * @dev Converts an unsigned uint256 into a signed int256.
     *
     * Requirements:
     *
     * - input must be less than or equal to maxInt256.
     */
    function toInt256(uint256 value) internal pure returns (int256) {
        // Note: Unsafe cast below is okay because `type(int256).max` is guaranteed to be positive
        if (value > uint256(type(int256).max)) {
            revert SafeCastOverflowedUintToInt(value);
        }
        return int256(value);
    }

    /**
     * @dev Cast a boolean (false or true) to a uint256 (0 or 1) with no jump.
     */
    function toUint(bool b) internal pure returns (uint256 u) {
        assembly ("memory-safe") {
            u := iszero(iszero(b))
        }
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/interface/IContractId.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

/**
 * @title IContractId
 * @notice Interface for contracts that have a contract ID identifier.
 * @dev This interface is separated from IUpgradeableVault to simplify override clauses
 *      in implementation contracts that inherit from multiple interfaces.
 */
interface IContractId {
    /**
     * @notice Get the contract's ID.
     * @return ID of the contract
     */
    function ID() external view returns (uint64);
}



// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/interface/IUpgradeableVault.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

import {IDefaultEnforcedVaultConfiguration} from "./IDefaultEnforcedVaultConfiguration.sol";

interface IUpgradeableVault {
    error AlreadyInitialized();
    error NotFactory();
    error NotInitialized();
    error InvalidFactoryOwner();
    error InvalidVaultImpl();

    /**
     * @notice Get the factory's address.
     * @return address of the factory
     */
    function factory() external view returns (address);

    /**
     * @notice Get the vault's Initialization version.
     * @return version of the vault (not necessarily the same as the Id of the vault)
     * @dev Starts from 1 and is incremented by 1 for each upgrade (except for legacy versions, which were upgraded also in larger increments)
     */
    function version() external view returns (uint64);

    /**
     * @notice Initialize `UpgradeableVaultProxy` contract by using a given data and setting a particular version and owner.
     * @param initialVersion initial version of the vault
     * @param owner initial owner of the vault
     * @param defaultEnforcedVaultConfig default configuration from the factory
     * @param data some data to use
     */
    function initialize(
        uint64 initialVersion,
        address owner,
        IDefaultEnforcedVaultConfiguration.DefaultEnforcedVaultConfiguration calldata defaultEnforcedVaultConfig,
        bytes calldata data
    ) external;

    /**
     * @notice Upgrade this vault to a specific newer version using a given data.
     * @param newVaultImpl new vault ID to upgrade to
     * @param data some data to use
     */
    function upgrade(uint64 newVaultImpl, bytes calldata data) external;
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/interface/IConcreteStandardVaultImpl.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

import {IUpgradeableVault} from "./IUpgradeableVault.sol";
import {IHurdleRateOracle} from "./IHurdleRateOracle.sol";
import {Hooks} from "./IHook.sol";
import {IERC4626} from "@openzeppelin-contracts/interfaces/IERC4626.sol";
import {IAccessControlEnumerable} from "@openzeppelin-contracts/access/extensions/IAccessControlEnumerable.sol";

/**
 * @title IConcreteStandardVaultImpl
 * @dev Interface for the standard vault implementation that manages multiple investment strategies.
 * @dev This interface extends the base tokenized vault functionality with strategy management capabilities.
 * @dev Strategies are external contracts that implement the IStrategyTemplate interface and handle
 * fund allocation to different yield-generating protocols or investment opportunities.
 */
interface IConcreteStandardVaultImpl is IUpgradeableVault, IERC4626, IAccessControlEnumerable {
    /**
     * @dev Thrown when attempting to withdraw to the zero address.
     */
    error InvalidReceiver();

    /**
     * @dev Thrown when attempting to add a strategy that uses a different asset than the vault.
     */
    error InvalidStrategyAsset();

    /**
     * @dev Thrown when attempting to add a strategy that is bound to a different vault.
     */
    error InvalidStrategyVault();

    /**
     * @dev Thrown when attempting to add a strategy that has already been added to the vault.
     */
    error StrategyAlreadyAdded();

    /**
     * @dev Thrown when attempting to operate on a strategy that doesn't exist in the vault.
     */
    error StrategyDoesNotExist();

    /**
     * @dev Thrown when attempting to interact with a strategy that is halted.
     */
    error StrategyIsHalted();

    /**
     * @dev Thrown when attempting to halt a strategy that is already halted.
     */
    error StrategyAlreadyHalted();

    /**
     * @dev Thrown when attempting to toggle the status of an inactive strategy.
     */
    error CannotToggleInactiveStrategy();

    /**
     * @dev Thrown when attempting to set a management fee without setting a recipient first.
     */
    error FeeRecipientNotSet();

    /**
     * @dev Thrown when attempting to set a management fee that exceeds the maximum allowed.
     */
    error ManagementFeeExceedsMaximum();

    /**
     * @dev Thrown when attempting to set a performance fee that exceeds the maximum allowed.
     */
    error PerformanceFeeExceedsMaximum();

    /**
     * @dev Thrown when attempting to set an invalid fee recipient address (address(0)).
     */
    error InvalidFeeRecipient();

    /**
     * @dev Thrown when the allocate module is invalid.
     */
    error InvalidAllocateModule();

    /**
     * @dev Thrown when the asset is invalid.
     */
    error InvalidAsset();

    /**
     * @dev Thrown when the initial vault manager is invalid.
     */
    error InvalidInitialVaultManager();

    /**
     * @dev Thrown when the name is invalid.
     */
    error InvalidName();

    /**
     * @dev Thrown when the symbol is invalid.
     */
    error InvalidSymbol();

    /**
     * @dev Thrown when the deposit limits are invalid.
     */
    error InvalidDepositLimits();

    /**
     * @dev Thrown when the withdraw limits are invalid.
     */
    error InvalidWithdrawLimits();

    /**
     * @dev Thrown when a hook's vault() does not match this vault.
     */
    error InvalidHookVault(address hook);

    /**
     *
     *
     **
     * @dev Thrown when the minimum deposit amount per transaction is not met.
     */
    error MinimumDepositAmountNotMet(address sender, uint256 assets, uint256 minTxDepositAmount);

    /**
     * @dev Thrown when the minimum withdraw amount per transaction is not met.
     */
    error MinimumWithdrawAmountNotMet(address sender, uint256 assets, uint256 minTxWithdrawAmount);

    /**
     * @dev Thrown when attempting to remove a strategy that has allocation or is in the deallocation order.
     */
    error StrategyHasAllocation();
    error StaleDeallocationOrder(address strategy);

    /**
     * @dev Thrown when the vault has insufficient balance to process the epoch.
     */
    error InsufficientBalance();

    /**
     * @dev Thrown when there are insufficient withdrawable assets available across all strategies.
     * @param requested The amount of assets requested for withdrawal.
     * @param available The total amount of assets available for withdrawal (idle + withdrawable from strategies).
     */
    error InsufficientWithdrawableAssets(uint256 requested, uint256 available);

    /**
     * @dev Thrown when calculated shares are zero.
     */
    error InsufficientShares();

    /**
     * @dev Thrown when calculated assets are zero.
     */
    error InsufficientAssets();

    /**
     * @dev Emitted when a new strategy is successfully added to the vault.
     * @param strategy The address of the strategy contract that was added.
     */
    event StrategyAdded(address strategy);

    /**
     * @dev Emitted when a strategy is successfully removed from the vault.
     * @param strategy The address of the strategy contract that was removed.
     */
    event StrategyRemoved(address strategy);

    /**
     * @dev Emitted when a strategy is set to Halted status.
     * @param strategy The address of the strategy contract that was halted.
     */
    event StrategyHalted(address strategy);

    /**
     * @dev Emitted when a strategy's status is toggled between Active and Halted.
     * @param strategy The address of the strategy contract whose status was toggled.
     */
    event StrategyStatusToggled(address indexed strategy);

    /**
     * @dev Emitted when the yield accrual operation is completed across all strategies.
     *
     * @param totalPositiveYield The total amount of positive yield generated across all strategies.
     * @param totalNegativeYield The total amount of losses incurred across all strategies.
     */
    event YieldAccrued(uint256 totalPositiveYield, uint256 totalNegativeYield);

    /**
     * @dev Emitted when management fee is accrued.
     * @param recipient The address that received the management fee shares.
     * @param shares The number of shares minted as management fee.
     * @param feeAmount The asset value of the management fee.
     */
    event ManagementFeeAccrued(address indexed recipient, uint256 shares, uint256 feeAmount);

    /**
     * @dev Emitted when performance fee is accrued.
     * @param recipient The address that received the performance fee shares.
     * @param shares The number of shares minted as performance fee.
     * @param feeAmount The asset value of the performance fee.
     */
    event PerformanceFeeAccrued(address indexed recipient, uint256 shares, uint256 feeAmount);

    /**
     * @dev Emitted when management fee is updated.
     * @param managementFee The new management fee rate in basis points.
     */
    event ManagementFeeUpdated(uint16 managementFee);

    /**
     * @dev Emitted when management fee recipient is updated.
     * @param managementFeeRecipient The new management fee recipient address.
     */
    event ManagementFeeRecipientUpdated(address managementFeeRecipient);

    /**
     * @dev Emitted when performance fee is updated.
     * @param performanceFee The new performance fee rate in basis points.
     */
    event PerformanceFeeUpdated(uint16 performanceFee);

    /**
     * @dev Emitted when performance fee recipient is updated.
     * @param performanceFeeRecipient The new performance fee recipient address.
     */
    event PerformanceFeeRecipientUpdated(address performanceFeeRecipient);

    /**
     * @dev Emitted when the hurdle rate oracle is updated.
     * @param oracle The new hurdle rate oracle address (address(0) disables hurdle gating).
     */
    event HurdleRateOracleUpdated(address indexed oracle);

    /**
     * @dev Emitted when deposit limits are updated.
     * @param maxGlobalDepositAmount The new maximum global deposit amount (applies to total assets across all users).
     * @param minTxDepositAmount The new minimum deposit amount per transaction.
     */
    event DepositLimitsUpdated(uint256 maxGlobalDepositAmount, uint256 minTxDepositAmount);

    /**
     * @dev Emitted when withdraw limits are updated.
     * @param maxTxWithdrawAmount The new maximum withdraw amount per transaction.
     * @param minTxWithdrawAmount The new minimum withdraw amount per transaction.
     */
    event WithdrawLimitsUpdated(uint256 maxTxWithdrawAmount, uint256 minTxWithdrawAmount);

    /**
     * @dev Emitted when the deallocation order is updated.
     */
    event DeallocationOrderUpdated();

    /**
     * @dev Emitted when an individual strategy's yield is accrued.
     *
     * @param strategy The address of the strategy contract whose yield was accrued.
     * @param currentTotalAllocatedValue The current total allocated value reported by the strategy.
     * @param yield The amount of positive yield generated by this strategy since last accrual.
     * @param loss The amount of loss incurred by this strategy since last accrual.
     */
    event StrategyYieldAccrued(
        address indexed strategy, uint256 currentTotalAllocatedValue, uint256 yield, uint256 loss
    );

    /**
     * @dev Enumeration of vault-manager–configurable parameters.
     * Used by {configure} to dispatch to the correct setter in a single call.
     */
    enum VaultManagerConfig {
        ManagementFee,
        PerformanceFee,
        HurdleRateOracle,
        DepositLimits,
        WithdrawLimits
    }

    error InvalidVaultManagerConfig();

    /**
     * @dev Enumeration of possible strategy statuses within the vault.
     * @dev Inactive: Strategy is inactive and cannot receive new allocations.
     * @dev Active: Strategy is active and can receive allocations and process withdrawals normally.
     * @dev Halted: Strategy is halted, typically due to detected issues or failures.
     * In this state, the strategy can be removed even if it has allocated funds
     */
    enum StrategyStatus {
        Inactive,
        Active,
        Halted
    }

    /**
     * @dev Structure containing metadata and state information for each strategy.
     * @dev status: Current operational status of the strategy.
     * @dev allocated: Total amount of vault assets currently allocated to this strategy, denominated in the vault's underlying asset token.
     */
    struct StrategyData {
        StrategyStatus status;
        uint120 allocated;
    }

    /**
     * @dev Adds a new strategy to the vault.
     * @dev The strategy must implement the IStrategyTemplate interface and use the same underlying asset as the vault.
     * @dev Only callable by accounts with the STRATEGY_MANAGER role.
     *
     * @param strategy The address of the strategy contract to add.
     *
     * Requirements:
     * - The strategy's asset() must match the vault's asset()
     * - The strategy must not already be added to the vault
     * - Caller must have STRATEGY_MANAGER role
     *
     * Emits:
     * - StrategyAdded event
     *
     * Reverts:
     * - InvalidStrategyAsset if strategy uses different asset
     * - StrategyAlreadyAdded if strategy is already in the vault
     */
    function addStrategy(address strategy) external;

    /**
     * @dev Removes a strategy from the vault.
     * @dev The strategy can only be removed if it has no allocated funds, unless it's in Halted status.
     * @dev Only callable by accounts with the STRATEGY_MANAGER role.
     *
     * @param strategy The address of the strategy contract to remove.
     *
     * Requirements:
     * - Strategy must exist in the vault
     * - Strategy must have zero allocated funds OR be in Halted status
     * - Caller must have STRATEGY_MANAGER role
     *
     * Emits:
     * - StrategyRemoved event
     *
     * Reverts:
     * - StrategyDoesNotExist if strategy is not in the vault
     * - Custom revert if strategy has allocated funds and is not in Halted status
     */
    function removeStrategy(address strategy) external;

    /**
     * @dev Toggles a strategy's status between Active and Halted.
     * @dev This is a safety mechanism to isolate problematic strategies or reactivate previously halted ones.
     * @dev Active strategies can receive allocations and participate in yield accrual and withdrawal operations.
     * @dev Halted strategies are skipped during yield accrual and withdrawal operations.
     * @dev Only callable by accounts with the STRATEGY_MANAGER role.
     *
     * @param strategy The address of the strategy contract to toggle.
     *
     * Requirements:
     * - Strategy must exist in the vault
     * - Strategy must be either Active or Halted (cannot toggle Inactive strategies)
     * - Caller must have STRATEGY_MANAGER role
     *
     * Emits:
     * - StrategyStatusToggled event
     *
     * Reverts:
     * - StrategyDoesNotExist if strategy is not in the vault
     */
    function toggleStrategyStatus(address strategy) external;

    /**
     * @notice Executes fund allocation and deallocation operations across multiple strategies.
     * @dev This function performs a yield accrual operation first to update vault accounting,
     *      then executes the allocation operations specified in the data parameter.
     * @dev All operations are performed via delegatecall to the respective modules to maintain
     *      proper storage context and access control.
     * @param data ABI-encoded array of AllocateParams structures containing the allocation
     *             operations to execute. Each param specifies whether to allocate or deallocate
     *             funds, which strategy to use, and any additional data required by the strategy.
     * @dev Only callable by accounts with the ALLOCATOR role.
     * @dev The function automatically triggers yield accrual before allocation to ensure
     *         accurate vault accounting prior to fund movements.
     */
    function allocate(bytes calldata data) external;

    /**
     * @notice Accrues yield and accounts for losses across all active strategies in the vault.
     * @dev This function updates the vault's internal accounting by querying the current
     *      value of all strategy allocations and calculating net yield or losses.
     * @dev This function can be called by anyone to update the vault's accounting.
     * @dev The yield accrual operation does not trigger actual fund movements, it only
     *         updates the vault's internal state to reflect current strategy values.
     */
    function accrueYield() external;

    /**
     * @notice Generic vault-manager setter that dispatches based on `configType`.
     * @dev Only callable by accounts with VAULT_MANAGER role.
     *      Applies nonReentrant + withYieldAccrual guards once for all config types.
     * @param configType The parameter to configure (see {VaultManagerConfig}).
     * @param data ABI-encoded value(s) for the chosen config type:
     *   - ManagementFee:    abi.encode(uint16 fee)
     *   - PerformanceFee:   abi.encode(uint16 fee)
     *   - HurdleRateOracle: abi.encode(IHurdleRateOracle oracle)
     *   - DepositLimits:    abi.encode(uint256 minTxDeposit, uint256 maxGlobalDeposit)
     *   - WithdrawLimits:   abi.encode(uint256 minTxWithdraw, uint256 maxTxWithdraw)
     */
    function configure(VaultManagerConfig configType, bytes calldata data) external;

    /**
     * @notice Updates the management fee recipient for the vault.
     * @param recipient The new management fee recipient address.
     * @dev Only callable by factory owner.
     * @dev Recipient cannot be address(0).
     */
    function updateManagementFeeRecipient(address recipient) external;

    /**
     * @notice Updates the performance fee recipient for the vault.
     * @param recipient The new performance fee recipient address.
     * @dev Only callable by factory owner.
     * @dev Recipient cannot be address(0).
     */
    function updatePerformanceFeeRecipient(address recipient) external;

    /**
     * @notice Returns the current fee configuration for the vault.
     * @return currentManagementFee The current management fee in basis points.
     * @return currentManagementFeeRecipient The current management fee recipient address.
     * @return currentLastManagementFeeAccrual The timestamp of the last management fee accrual.
     * @return currentPerformanceFee The current performance fee in basis points.
     * @return currentPerformanceFeeRecipient The current performance fee recipient address.
     */
    function getFeeConfig()
        external
        view
        returns (
            uint16 currentManagementFee,
            address currentManagementFeeRecipient,
            uint32 currentLastManagementFeeAccrual,
            uint16 currentPerformanceFee,
            address currentPerformanceFeeRecipient
        );

    /**
     * @notice Sets the hooks for the vault.
     * @dev This function sets the hooks for the vault.
     * @dev Only callable by accounts with the HOOK_MANAGER role.
     * @param hooks The hooks to set.
     */
    function setHooks(Hooks memory hooks) external;

    /**
     * @notice Returns the current hooks configuration for the vault.
     * @return The Hooks struct containing the target address and flags.
     */
    function getHooks() external view returns (Hooks memory);

    /**
     * @notice Previews the total assets that would be available after accruing yield from all strategies.
     * @dev This function simulates the yield accrual operation without actually executing it,
     *      providing a view of what the vault's total assets would be after accounting
     *      for yield and losses across all active strategies.
     * @dev The calculation includes the current lastTotalAssets plus any positive
     *      yield minus any losses that would be realized during yield accrual.
     * @dev This is a view function that does not modify state or trigger any actual
     *      fund movements or strategy interactions.
     *
     * @return The total amount of assets that would be available in the vault after yield accrual,
     *         denominated in the vault's underlying asset token.
     * @return The total amount of shares that would be available in the vault after yield accrual,
     *         calculated as current totalSupply + management fee shares.
     */
    function previewAccrueYield() external view returns (uint256, uint256);

    /**
     * @dev Retrieves the current data and status information for a specific strategy.
     * @dev This function provides read-only access to strategy metadata including allocation amounts and status.
     *
     * @param strategy The address of the strategy contract to query.
     * @return The StrategyData struct containing the strategy's current status and allocated amount.
     *
     * Note:
     * - Returns default values (Inactive status, 0 allocated) for non-existent strategies
     * - Does not revert for invalid strategy addresses
     */
    function getStrategyData(address strategy) external view returns (StrategyData memory);

    /**
     * @dev Returns an array of all strategy addresses currently managed by the vault.
     * @dev This function provides a way to enumerate all active strategies for external integrations and monitoring.
     *
     * @return An array containing the addresses of all strategies added to the vault.
     *
     * Note:
     * - The returned array includes strategies in all statuses (Active, Inactive, Emergency)
     * - The order of strategies in the array is not guaranteed
     * - Returns an empty array if no strategies have been added
     */
    function getStrategies() external view returns (address[] memory);

    /**
     * @dev Returns the address of the allocate module.
     *
     * @return The address of the allocate module.
     */
    function allocateModule() external view returns (address);

    /**
     * @dev Returns the deposit limits.
     * @return maxGlobalDepositAmount The maximum deposit amount for the vault.
     * @return minTxDepositAmount The minimum deposit amount for a single transaction.
     */
    function getDepositLimits() external view returns (uint256 maxGlobalDepositAmount, uint256 minTxDepositAmount);

    /**
     * @dev Returns the withdraw limits.
     * @return maxTxWithdrawAmount The maximum withdraw amount for a single transaction.
     * @return minTxWithdrawAmount The minimum withdraw amount for a single transaction.
     */
    function getWithdrawLimits() external view returns (uint256 maxTxWithdrawAmount, uint256 minTxWithdrawAmount);

    /**
     * @dev Returns the total amount of assets allocated to all strategies.
     *
     * @return The total amount of assets allocated to all strategies.
     */
    function getTotalAllocated() external view returns (uint256);

    /**
     * @dev Returns the cached value of total assets after the last call.
     *
     * @return The cached value of total assets after the last call.
     */
    function cachedTotalAssets() external view returns (uint256);

    /**
     * @dev Returns the deallocation order from strategies.
     *
     * @return order An array of strategy addresses in the order they should be deallocated.
     */
    function getDeallocationOrder() external view returns (address[] memory order);

    /**
     * @dev Sets the deallocation order for strategies.
     * @dev Only callable by accounts with the ALLOCATOR role.
     * @param order An array of strategy addresses in the order they should be deallocated.
     */
    function setDeallocationOrder(address[] calldata order) external;

    /**
     * @notice Returns the current hurdle rate oracle address.
     * @return The hurdle rate oracle, or address(0) if not set.
     */
    function getHurdleRateOracle() external view returns (IHurdleRateOracle);
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/interface/IDefaultEnforcedVaultConfiguration.sol =====
// SPDX-License-Identifier: AGPL-3.0
pragma solidity ^0.8.24;

/// @title IDefaultEnforcedVaultConfiguration
/// @notice Interface defining the default configuration enforced by the protocol
interface IDefaultEnforcedVaultConfiguration {
    struct DefaultEnforcedVaultConfiguration {
        /// @dev Default Management fee recipient enforced by the protocol
        address defaultManagementFeeRecipient;
        /// @dev Default Performance fee recipient enforced by the protocol
        address defaultPerformanceFeeRecipient;
    }
}



// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/interface/IHurdleRateOracle.sol =====
// SPDX-License-Identifier: AGPL-3.0
pragma solidity ^0.8.24;

/**
 * @title IHurdleRateOracle
 * @notice Oracle that returns the expected exchange rate (assets per share) that the
 *         vault must exceed before performance fees are charged.
 * @dev The hurdle rate is expressed as an exchange rate scaled to `precision()`.
 *      For example, with `precision() == 1e18`, a hurdle rate of 1.05e18 means the vault
 *      must reach 1.05 assets per share before performance fees activate.
 *      Implementations may compute this on-chain (e.g. linear/compound growth from a
 *      baseline) or relay an off-chain computed value. Each oracle instance is bound
 *      to a single vault.
 */
interface IHurdleRateOracle {
    /**
     * @notice Returns the scaling factor for the hurdle rate.
     * @dev The vault divides by this value when converting the hurdle exchange rate
     *      to asset amounts: `hurdleAssets = hurdleRate * totalSupply / precision()`.
     */
    function precision() external view returns (uint256);

    /**
     * @notice Returns the current hurdle rate at the current block timestamp.
     * @return hurdleRate The minimum assets-per-share value below which no
     *         performance fee should be charged.
     */
    function getHurdleRate() external view returns (uint256 hurdleRate);
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/interface/IStrategyTemplate.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

/**
 * @title IStrategyTemplate
 * @dev Interface that all strategy implementations must follow to be compatible with the vault system.
 * @dev Each strategy is bound to a single vault and manages that vault's funds in different protocols or investment opportunities.
 * @dev The Vault uses this interface to deploy, withdraw, and rebalance funds across multiple strategies.
 *
 * @notice This interface defines the core functionality required for strategy contracts:
 * - Asset management (allocation and deallocation of funds)
 * - Withdrawal capabilities for user redemptions
 * - Limit reporting for rebalancing operations
 * - Compatibility with the vault's underlying asset token
 *
 * @notice All strategies must implement proper access controls and ensure only authorized callers
 * (typically the vault) can execute fund management operations.
 *
 * @notice For strategies that accrue rewards from underlying protocols:
 * The vault has an arbitrary call execution function that can call any target with arbitrary data.
 * This is primarily used to claim rewards from external reward systems. Strategies that earn rewards
 * should provide dedicated functions that can be called by the vault through this mechanism to claim
 * rewards and forward them to the rewards distributor system.
 */

/**
 * @dev Enum representing different types of strategies
 */
enum StrategyType {
    ATOMIC, // 0: Strategy that executes operations atomically, provides on-chain accurate accounting of yield
    ASYNC, // 1: Strategy that requires asynchronous operations (multiple transactions), can provide stale (within defined latency) accounting of yield
    CROSSCHAIN // 2: Strategy that operates across different blockchain networks, can provide stale (within defined latency) accounting of yield
}

interface IStrategyTemplate {
    /**
     * @dev Allocates funds from the vault to the underlying protocol.
     * @dev This function will be called when the vault wants to deploy assets into the yield-generating protocol.
     *
     * @param data Arbitrary calldata that can be used to pass strategy-specific parameters for the allocation.
     *             This allows for flexible configuration of the allocation process (e.g., slippage tolerance,
     *             specific protocol parameters, routing information, etc.).
     *
     * - MUST emit the AllocateFunds event.
     * - MUST revert if all of assets cannot be deposited (due to allocation limit being reached, slippage, the protocol
     *   not being able to accept more funds, etc).
     *
     * NOTE: most implementations will require pre-approval of the Vault with the Vault's underlying asset token.
     */
    function allocateFunds(bytes calldata data) external returns (uint256);

    /**
     * @dev Deallocates funds from the underlying protocol back to the vault.
     * @dev This function will be called when the vault wants to withdraw assets from the yield-generating protocol.
     *
     * @param data Arbitrary calldata that can be used to pass strategy-specific parameters for the deallocation.
     *             This allows for flexible configuration of the withdrawal process (e.g., slippage tolerance,
     *             specific protocol parameters, withdrawal routing, etc.).
     *
     * - MUST emit the DeallocateFunds event.
     * - MUST revert if all of assets cannot be withdrawn (due to withdrawal limit being reached, slippage, the protocol
     *   not having enough liquidity, etc).
     */
    function deallocateFunds(bytes calldata data) external returns (uint256);

    /**
     * @dev Sends assets of underlying tokens to sender.
     * @dev This function will be called when the vault unwinds its position while depositor withdraws.
     *
     * - MUST emit the Withdraw event.
     * - MUST revert if all of assets cannot be withdrawn (due to withdrawal limit being reached, slippage, the owner
     *   not having enough assets, etc).
     */
    function onWithdraw(uint256 assets) external returns (uint256);

    /**
     * @dev Rescue function to withdraw tokens that may have been accidentally sent to the strategy.
     * @dev This function allows authorized users to rescue tokens that are not part of the strategy's normal operations.
     *
     * @param token The address of the token to rescue.
     * @param amount The amount of tokens to rescue. Use 0 to rescue all available tokens.
     *
     * - MUST only allow rescue of tokens that are not the strategy's primary asset (asset()).
     * - MUST emit appropriate events for the rescue operation.
     * - MUST revert if the caller is not authorized to perform token rescue.
     * - MUST revert if attempting to rescue the strategy's primary asset token.
     */
    function rescueToken(address token, uint256 amount) external;

    /**
     * @dev Returns the address of the underlying token used for the Vault for accounting, depositing, and withdrawing.
     *
     * - MUST be an ERC-20 token contract.
     * - MUST NOT revert.
     */
    function asset() external view returns (address);

    /**
     * @dev Returns the address of the vault that this strategy is bound to.
     *
     * - MUST return the vault address that was set during strategy initialization.
     * - MUST NOT revert.
     */
    function getVault() external view returns (address);

    /**
     * @dev Returns the type of strategy implementation.
     * @dev This function indicates the operational characteristics of the strategy.
     *
     * @return The strategy type as defined in the StrategyType enum.
     *
     * - MUST return one of the defined StrategyType values.
     * - MUST NOT revert.
     * - ATOMIC: Strategy executes operations atomically in the same transaction, yield MUST be always atomicly updated in strategy allocated amount.
     * - ASYNC: Strategy requires asynchronous operations across multiple transactions, yield Can be updated asynchronously within documented latency.
     * - CROSSCHAIN: Strategy operates across different blockchain networks, yield Can be updated asynchronously within documented latency.
     */
    function strategyType() external view returns (StrategyType);

    /**
     * @dev Returns the total value of assets that the bound vault has allocated in the strategy.
     * @dev This function is mainly used during yield accrual operations to account for strategy yield or losses.
     *
     * @return The total value of allocated assets denominated in the asset() token.
     *
     * - MUST return the total value of assets that the bound vault has allocated to this strategy.
     * - MUST account for any losses or depreciation in the underlying protocol.
     * - MUST NOT revert.
     * - MUST return 0 if the vault has no funds allocated to this strategy.
     */
    function totalAllocatedValue() external view returns (uint256);

    /**
     * @dev Returns the maximum amount of assets that can be allocated to the underlying protocol.
     * @dev This function is primarily used by the Allocator to determine allocation limits when rebalancing funds.
     *
     * - MUST return the maximum amount of underlying assets that can be allocated in a single call to allocateFunds.
     * - MUST NOT revert.
     * - MAY return 0 if the protocol cannot accept any more funds.
     * - MAY return type(uint256).max if there is no practical limit.
     */
    function maxAllocation() external view returns (uint256);

    /**
     * @dev Returns the maximum amount of assets that can be withdrawn from the strategy by the vault.
     * @dev This function is primarily used by the vault to determine withdrawal limits when covering user redemptions.
     *
     * - MUST return the maximum amount of underlying assets that can be withdrawn in a single call to onWithdraw.
     * - MUST NOT revert.
     * - MAY return 0 if no funds are available for withdrawal.
     * - SHOULD reflect current liquidity constraints and strategy-specific withdrawal limits.
     */
    function maxWithdraw() external view returns (uint256);
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/module/AllocateModule.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

import {IERC20, IERC4626} from "@openzeppelin-contracts/interfaces/IERC4626.sol";
import {IAllocateModule} from "../interface/IAllocateModule.sol";
import {IStrategyTemplate} from "../interface/IStrategyTemplate.sol";
import {IConcreteStandardVaultImpl} from "../interface/IConcreteStandardVaultImpl.sol";
import {ConcreteStandardVaultImplStorageLib} from "../lib/storage/ConcreteStandardVaultImplStorageLib.sol";
import {SafeCast} from "@openzeppelin-contracts/utils/math/SafeCast.sol";
import {SafeERC20} from "@openzeppelin-contracts/token/ERC20/utils/SafeERC20.sol";

contract AllocateModule is IAllocateModule {
    using SafeCast for uint256;
    using SafeERC20 for IERC20;

    /// @inheritdoc IAllocateModule
    function allocateFunds(bytes calldata data) external {
        AllocateParams[] memory params = abi.decode(data, (AllocateParams[]));

        ConcreteStandardVaultImplStorageLib.ConcreteStandardVaultImplStorage storage $ =
            ConcreteStandardVaultImplStorageLib.fetch();

        for (uint256 i; i < params.length; ++i) {
            // Only allocate to Active strategies
            if ($.strategyData[params[i].strategy].status != IConcreteStandardVaultImpl.StrategyStatus.Active) {
                continue;
            }

            uint256 amount;
            if (params[i].isDeposit) {
                IERC20(IERC4626(address(this)).asset()).forceApprove(params[i].strategy, type(uint256).max);

                amount = IStrategyTemplate(params[i].strategy).allocateFunds(params[i].extraData);

                IERC20(IERC4626(address(this)).asset()).forceApprove(params[i].strategy, 0);

                $.strategyData[params[i].strategy].allocated += amount.toUint120();
            } else {
                amount = IStrategyTemplate(params[i].strategy).deallocateFunds(params[i].extraData);
                $.strategyData[params[i].strategy].allocated -= amount.toUint120();
            }

            emit AllocatedFunds(params[i].strategy, params[i].isDeposit, amount, params[i].extraData);
        }
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/common/UpgradeableVault.sol =====
// SPDX-License-Identifier: AGPL-3.0
pragma solidity ^0.8.24;

/**
 * @title AbstractUpgradeableVault
 * @notice Abstract upgradeable base for Concrete vault implementations.
 *         Provides initializer pattern, ownership, and reentrancy guards.
 *
 * @author Blueprint Finance
 * @custom:protocol Concrete Earn V2
 * @custom:source on request
 * @custom:audits on request
 * @custom:license  AGPL-3.0
 */

// ─────────────────────────────────────────────────────────────────────────────
// External dependencies
// ─────────────────────────────────────────────────────────────────────────────
import {Initializable} from "@openzeppelin-upgradeable/proxy/utils/Initializable.sol";
import {OwnableUpgradeable} from "@openzeppelin-upgradeable/access/OwnableUpgradeable.sol";
import {ReentrancyGuardTransient} from "@openzeppelin-contracts/utils/ReentrancyGuardTransient.sol";

// ─────────────────────────────────────────────────────────────────────────────
// Protocol-facing interfaces
// ─────────────────────────────────────────────────────────────────────────────
import {IDefaultEnforcedVaultConfiguration} from "../interface/IDefaultEnforcedVaultConfiguration.sol";
import {IUpgradeableVault} from "../interface/IUpgradeableVault.sol";

// ─────────────────────────────────────────────────────────────────────────────
// Storage layout libraries
// ─────────────────────────────────────────────────────────────────────────────
abstract contract UpgradeableVault is Initializable, OwnableUpgradeable, ReentrancyGuardTransient, IUpgradeableVault {
    address public immutable factory;

    /**
     * @notice Get the contract's ID.
     * @return ID of the contract
     * @dev This is a virtual function that must be overridden by concrete implementations.
     *      Concrete implementations should also implement IContractId interface.
     */
    function ID() public pure virtual returns (uint64);

    constructor(address factoryAddr) {
        _disableInitializers();

        factory = factoryAddr;
    }

    modifier notInitialized() {
        if (_getInitializedVersion() != 0) {
            revert AlreadyInitialized();
        }

        _;
    }

    /**
     * @inheritdoc IUpgradeableVault
     */
    function version() public view returns (uint64) {
        return _getInitializedVersion();
    }

    /**
     * @inheritdoc IUpgradeableVault
     */
    function initialize(
        uint64 initialVaultImpl,
        address owner_,
        IDefaultEnforcedVaultConfiguration.DefaultEnforcedVaultConfiguration calldata defaultEnforcedVaultConfig,
        bytes calldata data
    ) external notInitialized initializer {
        require(_msgSender() == factory, NotFactory());
        require(initialVaultImpl == ID(), InvalidVaultImpl());

        __Ownable_init(owner_);

        _initialize(initialVaultImpl, owner_, defaultEnforcedVaultConfig, data);
    }

    /**
     * @inheritdoc IUpgradeableVault
     */
    function upgrade(uint64 newVaultImpl, bytes calldata data)
        external
        nonReentrant
        reinitializer(_getInitializedVersion() + 1)
    {
        require(_msgSender() == factory, NotFactory());
        require(newVaultImpl == ID(), InvalidVaultImpl());
        _upgrade(newVaultImpl, data);
    }

    /**
     *
     * @param initialVaultImpl initial vault ID from the factory
     * @param owner vault proxy owner address
     * @param defaultEnforcedVaultConfig default configuration from the factory
     * @param data arbitrary data used to initialize a proxy implementation
     */
    function _initialize(
        uint64 initialVaultImpl,
        address owner,
        IDefaultEnforcedVaultConfiguration.DefaultEnforcedVaultConfiguration calldata defaultEnforcedVaultConfig,
        bytes memory data
    ) internal virtual {}

    /**
     *
     * @param newVaultImpl vault proxy new vault ID
     * @param data arbitrary data that will be used on the new `newImplementation.upgrade()` function to execute the upgrade flow
     */
    function _upgrade(uint64 newVaultImpl, bytes calldata data) internal virtual {}
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/Conversion.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Math} from "@openzeppelin-contracts/utils/math/Math.sol";
import {IConcreteStandardVaultImpl} from "../interface/IConcreteStandardVaultImpl.sol";

library ConcreteV2ConversionLib {
    using Math for uint256;

    function calcConvertToShares(
        uint256 assets,
        uint256 _totalSupply,
        uint256 _totalAssets,
        Math.Rounding rounding,
        bool safeMode
    ) internal pure returns (uint256 shares) {
        // setting uint256 decimalsOffset = 0;
        shares = assets.mulDiv(
            _totalSupply + 1, // + 10 ** decimalsOffset = 1
            _totalAssets + 1,
            rounding
        );
        if (safeMode && shares == 0) revert IConcreteStandardVaultImpl.InsufficientShares();
    }

    function calcConvertToAssets(
        uint256 shares,
        uint256 _totalSupply,
        uint256 _totalAssets,
        Math.Rounding rounding,
        bool safeMode
    ) internal pure returns (uint256 assets) {
        // setting uint256 decimalsOffset = 0;
        assets = shares.mulDiv(
            _totalAssets + 1,
            _totalSupply + 1, // + 10 ** decimalsOffset = 1
            rounding
        );
        if (safeMode && assets == 0) revert IConcreteStandardVaultImpl.InsufficientAssets();
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/StandardVaultHelperLib.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Math} from "@openzeppelin-contracts/utils/math/Math.sol";
import {ConcreteV2ConversionLib as ConversionLib} from "./Conversion.sol";
import {ConcreteStandardVaultImplStorageLib as SVLib} from "./storage/ConcreteStandardVaultImplStorageLib.sol";
import {EnumerableSet} from "@openzeppelin-contracts/utils/structs/EnumerableSet.sol";

library StandardVaultHelperLib {
    using EnumerableSet for EnumerableSet.AddressSet;

    /**
     * @dev Calculates the maximum deposit amount based on deposit limits and current total assets.
     * @param maxGlobalDepositLimit The maximum global deposit limit configured for the vault (applies to total assets across all users).
     * @param minTxDepositAmount The minimum deposit amount configured for the vault.
     * @param totalAssets_ The current total assets (after yield accrual).
     * @return The maximum amount of assets that can be deposited, or 0 if deposits are disabled or limit reached.
     */
    function calcMaxDeposit(uint256 maxGlobalDepositLimit, uint256 minTxDepositAmount, uint256 totalAssets_)
        internal
        pure
        returns (uint256)
    {
        // If deposits are disabled, return 0
        if (maxGlobalDepositLimit == 0) {
            return 0;
        }

        // If limit already reached, return 0
        if (totalAssets_ >= maxGlobalDepositLimit) {
            return 0;
        }

        uint256 maxRemainingDepositAmount = maxGlobalDepositLimit - totalAssets_;
        // Calculate maximum available deposit amount
        return maxRemainingDepositAmount >= minTxDepositAmount ? maxRemainingDepositAmount : 0;
    }

    /**
     * @dev Calculates the maximum mint amount (in shares) based on deposit limits and current vault state.
     * @param maxGlobalDepositLimit The maximum global deposit limit configured for the vault (applies to total assets across all users).
     * @param minTxDepositAmount The minimum deposit amount configured for the vault.
     * @param totalAssets_ The current total assets (after yield accrual).
     * @param totalSupply_ The current total supply (after yield accrual).
     * @return The maximum amount of shares that can be minted, or 0 if deposits are disabled or limit reached.
     */
    function calcMaxMint(
        uint256 maxGlobalDepositLimit,
        uint256 minTxDepositAmount,
        uint256 totalAssets_,
        uint256 totalSupply_
    ) internal pure returns (uint256) {
        uint256 maxDepositAmount = calcMaxDeposit(maxGlobalDepositLimit, minTxDepositAmount, totalAssets_);

        // Early return if no deposit is allowed
        if (maxDepositAmount == 0) {
            return 0;
        }

        return
            ConversionLib.calcConvertToShares(maxDepositAmount, totalSupply_, totalAssets_, Math.Rounding.Floor, false);
    }

    /**
     * @dev Calculates the maximum withdraw amount (in assets) based on owner balance and withdraw limits.
     * @param ownerShares The amount of shares owned by the user.
     * @param maxTxWithdrawLimit The maximum withdraw limit configured for the vault.
     * @param minTxWithdrawAmount The minimum withdraw amount configured for the vault.
     * @param totalAssets_ The current total assets (after yield accrual).
     * @param totalSupply_ The current total supply (after yield accrual).
     * @return The maximum amount of assets that can be withdrawn, or 0 if withdrawals are disabled.
     */
    function calcMaxWithdraw(
        uint256 ownerShares,
        uint256 maxTxWithdrawLimit,
        uint256 minTxWithdrawAmount,
        uint256 totalAssets_,
        uint256 totalSupply_
    ) internal pure returns (uint256) {
        // If withdrawals are disabled, return 0
        if (maxTxWithdrawLimit == 0) {
            return 0;
        }

        // Calculate owner's share of assets
        uint256 ownerAssets =
            ConversionLib.calcConvertToAssets(ownerShares, totalSupply_, totalAssets_, Math.Rounding.Floor, false);

        uint256 maxWithdrawAmount = ownerAssets > maxTxWithdrawLimit ? maxTxWithdrawLimit : ownerAssets;
        // Return the minimum between owner's assets and the max withdraw limit
        return maxWithdrawAmount >= minTxWithdrawAmount ? maxWithdrawAmount : 0;
    }

    /**
     * @dev Calculates the maximum redeem amount (in shares) based on owner balance and withdraw limits.
     * @param ownerShares The amount of shares owned by the user.
     * @param maxTxWithdrawLimit The maximum withdraw limit configured for the vault.
     * @param minTxWithdrawAmount The minimum withdraw amount configured for the vault.
     * @param totalAssets_ The current total assets (after yield accrual).
     * @param totalSupply_ The current total supply (after yield accrual).
     * @return The maximum amount of shares that can be redeemed, or 0 if withdrawals are disabled.
     * @dev This function converts maxTxWithdrawLimit directly to shares (rounding down conservatively)
     *      to avoid double rounding errors that could prevent users from redeeming all their shares.
     */
    function calcMaxRedeem(
        uint256 ownerShares,
        uint256 maxTxWithdrawLimit,
        uint256 minTxWithdrawAmount,
        uint256 totalAssets_,
        uint256 totalSupply_
    ) internal pure returns (uint256) {
        // If withdrawals are disabled, return 0
        if (maxTxWithdrawLimit == 0) {
            return 0;
        }

        uint256 maxShares;

        if (maxTxWithdrawLimit == type(uint256).max) {
            maxShares = ownerShares;
        } else {
            // Convert maxTxWithdrawLimit directly to shares (rounding down conservatively)
            // This avoids double rounding: ownerShares -> ownerAssets -> maxShares
            // Instead: maxTxWithdrawLimit -> maxShares, then compare directly with ownerShares
            uint256 maxSharesFromLimit = ConversionLib.calcConvertToShares(
                maxTxWithdrawLimit, totalSupply_, totalAssets_, Math.Rounding.Floor, false
            );
            maxShares = ownerShares > maxSharesFromLimit ? maxSharesFromLimit : ownerShares;
        }

        uint256 resultingAssets =
            ConversionLib.calcConvertToAssets(maxShares, totalSupply_, totalAssets_, Math.Rounding.Floor, false);
        return resultingAssets >= minTxWithdrawAmount ? maxShares : 0;
    }

    /**
     * @dev Calculates the total amount of assets allocated across all strategies.
     * @return totalAllocated The sum of all strategy allocations.
     */
    function calcTotalAllocated() public view returns (uint256 totalAllocated) {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        address[] memory strategies = $.strategies.values();
        uint256 strategiesLength = strategies.length;

        for (uint256 i; i < strategiesLength; ++i) {
            totalAllocated += $.strategyData[strategies[i]].allocated;
        }
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/Hooks.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Hooks, IHook} from "../interface/IHook.sol";

error FlagIndexOutOfBounds(uint8 flagIndex);

/// @title HooksLibV2
/// @dev Hooks library for user action hooks (pre/post deposit, mint, withdraw, redeem),
///      strategy hooks (pre-add-strategy, pre-remove-strategy),
///      and transfer hooks (pre/post transfer).
library HooksLibV2 {
    uint8 constant PRE_DEPOSIT = 1;
    uint8 constant POST_DEPOSIT = 2;
    uint8 constant PRE_MINT = 3;
    uint8 constant POST_MINT = 4;
    uint8 constant PRE_WITHDRAW = 5;
    uint8 constant POST_WITHDRAW = 6;
    uint8 constant PRE_REDEEM = 7;
    uint8 constant POST_REDEEM = 8;

    uint8 constant PRE_ADD_STRATEGY = 9;
    uint8 constant PRE_REMOVE_STRATEGY = 10;

    uint8 constant PRE_TRANSFER = 11;
    uint8 constant POST_TRANSFER = 12;

    /// @dev Checks if a specific flag is set in the Hooks struct
    /// @param h The Hooks storage reference
    /// @param flagIndex The flag index to check (0-95)
    /// @return True if the flag is set, false otherwise
    function flagIsSet(Hooks memory h, uint8 flagIndex) internal pure returns (bool) {
        if (flagIndex >= 96) return false;
        return (uint96(h.flags) & (1 << flagIndex)) != 0;
    }

    function setFlag(Hooks memory h, uint8 flagIndex) internal pure returns (uint96) {
        if (flagIndex >= 96) revert FlagIndexOutOfBounds(flagIndex);
        h.flags = uint96(h.flags | (1 << flagIndex));
        return h.flags;
    }

    function checkIsValid(Hooks memory h, uint8 flagIndex) internal pure returns (bool) {
        if (!flagIsSet(h, flagIndex) || h.target == address(0)) return false;
        return true;
    }

    function preDeposit(
        Hooks memory h,
        address sender,
        uint256 assets,
        uint256 shares,
        address receiver,
        uint256 totalAssets
    ) internal {
        IHook(h.target).preDeposit(sender, assets, shares, receiver, totalAssets);
    }

    function preMint(
        Hooks memory h,
        address sender,
        uint256 assets,
        uint256 shares,
        address receiver,
        uint256 totalAssets
    ) internal {
        IHook(h.target).preMint(sender, assets, shares, receiver, totalAssets);
    }

    function preWithdraw(
        Hooks memory h,
        address sender,
        uint256 assets,
        uint256 shares,
        address receiver,
        address owner,
        uint256 totalAssets
    ) internal {
        IHook(h.target).preWithdraw(sender, assets, shares, receiver, owner, totalAssets);
    }

    function preRedeem(
        Hooks memory h,
        address sender,
        uint256 assets,
        uint256 shares,
        address receiver,
        address owner,
        uint256 totalAssets
    ) internal {
        IHook(h.target).preRedeem(sender, assets, shares, receiver, owner, totalAssets);
    }

    function postDeposit(
        Hooks memory h,
        address sender,
        uint256 assets,
        uint256 shares,
        address receiver,
        uint256 totalAssets
    ) internal {
        IHook(h.target).postDeposit(sender, assets, shares, receiver, totalAssets);
    }

    function postMint(
        Hooks memory h,
        address sender,
        uint256 assets,
        uint256 shares,
        address receiver,
        uint256 totalAssets
    ) internal {
        IHook(h.target).postMint(sender, assets, shares, receiver, totalAssets);
    }

    function postWithdraw(
        Hooks memory h,
        address sender,
        uint256 assets,
        uint256 shares,
        address receiver,
        uint256 totalAssets
    ) internal {
        IHook(h.target).postWithdraw(sender, assets, shares, receiver, totalAssets);
    }

    function postRedeem(
        Hooks memory h,
        address sender,
        uint256 assets,
        uint256 shares,
        address receiver,
        uint256 totalAssets
    ) internal {
        IHook(h.target).postRedeem(sender, assets, shares, receiver, totalAssets);
    }

    function preTransfer(Hooks memory h, address sender, address from, address to, uint256 shares) internal {
        IHook(h.target).preTransfer(sender, from, to, shares);
    }

    function postTransfer(Hooks memory h, address sender, address from, address to, uint256 shares) internal {
        IHook(h.target).postTransfer(sender, from, to, shares);
    }
}



// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/interface/IHook.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

struct Hooks {
    address target;
    uint96 flags;
}

interface IHook {
    function vault() external view returns (address);
    // USER ACTION HOOKS
    function preDeposit(address sender, uint256 assets, uint256 shares, address receiver, uint256 totalAssets) external;
    function preMint(address sender, uint256 assets, uint256 shares, address receiver, uint256 totalAssets) external;
    function preWithdraw(
        address sender,
        uint256 assets,
        uint256 shares,
        address receiver,
        address owner,
        uint256 totalAssets
    ) external;
    function preRedeem(
        address sender,
        uint256 assets,
        uint256 shares,
        address receiver,
        address owner,
        uint256 totalAssets
    ) external;
    function postDeposit(address sender, uint256 assets, uint256 shares, address receiver, uint256 totalAssets) external;
    function postMint(address sender, uint256 assets, uint256 shares, address receiver, uint256 totalAssets) external;
    function postWithdraw(address sender, uint256 assets, uint256 shares, address receiver, uint256 totalAssets)
        external;
    function postRedeem(address sender, uint256 assets, uint256 shares, address receiver, uint256 totalAssets) external;

    // TRANSFER HOOKS
    function preTransfer(address sender, address from, address to, uint256 shares) external;
    function postTransfer(address sender, address from, address to, uint256 shares) external;
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/Roles.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

library ConcreteV2RolesLib {
    /// Roles
    bytes32 public constant VAULT_MANAGER = keccak256("VAULT_MANAGER");
    bytes32 public constant HOOK_MANAGER = keccak256("HOOK_MANAGER");
    bytes32 public constant STRATEGY_MANAGER = keccak256("STRATEGY_MANAGER");
    bytes32 public constant ALLOCATOR = keccak256("ALLOCATOR");
    bytes32 public constant WITHDRAWAL_MANAGER = keccak256("WITHDRAWAL_MANAGER");
    bytes32 public constant PAUSER = keccak256("PAUSER");
    bytes32 public constant PRIORITY_WITHDRAWAL_EXECUTOR = keccak256("PRIORITY_WITHDRAWAL_EXECUTOR");

    /// Single role admin for all roles
    bytes32 public constant ROLE_ADMIN = keccak256("ROLE_ADMIN");
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/common/ERC20UpgradeableWithTransferHook.sol =====
// SPDX-License-Identifier: AGPL-3.0
pragma solidity ^0.8.24;

import {ERC20Upgradeable} from "@openzeppelin-upgradeable/token/ERC20/ERC20Upgradeable.sol";
import {HooksLibV2 as HooksLib, Hooks} from "../lib/Hooks.sol";
import {ConcreteStandardVaultImplStorageLib as SVLib} from "../lib/storage/ConcreteStandardVaultImplStorageLib.sol";

/**
 * @title ERC20UpgradeableWithTransferHook
 * @notice Mixin that injects pre/post transfer hook calls into the ERC20 `_update` flow.
 * @dev Only fires on actual transfers (both `from` and `to` are non-zero).
 *      Mints (`from == address(0)`) and burns (`to == address(0)`) bypass hooks entirely,
 *      as those operations have their own dedicated hook points in the vault.
 * @dev IMPORTANT: Reads hook configuration from ConcreteStandardVaultImplStorageLib.
 *      Only usable within the ConcreteStandardVaultImpl inheritance hierarchy.
 */
abstract contract ERC20UpgradeableWithTransferHook is ERC20Upgradeable {
    using HooksLib for Hooks;

    function _update(address from, address to, uint256 value) internal virtual override(ERC20Upgradeable) {
        if (from != address(0) && to != address(0)) {
            Hooks memory h = SVLib.fetch().hooks;
            if (h.checkIsValid(HooksLib.PRE_TRANSFER)) {
                h.preTransfer(_msgSender(), from, to, value);
            }
            ERC20Upgradeable._update(from, to, value);
            if (h.checkIsValid(HooksLib.POST_TRANSFER)) {
                h.postTransfer(_msgSender(), from, to, value);
            }
        } else {
            ERC20Upgradeable._update(from, to, value);
        }
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/StateInitLib.sol =====
// SPDX-License-Identifier: AGPL-3.0
pragma solidity ^0.8.24;

import {ACL_OZ_5_2_0_Lib} from "./AccessControlLib.sol";
import {ConcreteV2RolesLib as RolesLib} from "../lib/Roles.sol";
import {ConcreteStandardVaultImplStorageLib as SVLib} from "../lib/storage/ConcreteStandardVaultImplStorageLib.sol";
import {ConcreteAsyncVaultImplStorageLib as AVLib} from "../lib/storage/ConcreteAsyncVaultImplStorageLib.sol";
import {IConcreteAsyncVaultImpl} from "../interface/IConcreteAsyncVaultImpl.sol";
import {RoleMigrationLib} from "./RoleMigrationLib.sol";
import {Time} from "./Time.sol";

library StateInitLib {
    function stateInitStandardVaultImpl(address allocateModuleAddr, address initialVaultManager, address sender)
        public
    {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        $.allocateModule = allocateModuleAddr;
        $.lastManagementFeeAccrual = Time.timestamp();

        $.maxGlobalDepositAmount = type(uint256).max;
        $.minTxDepositAmount = 0;
        $.maxTxWithdrawAmount = type(uint256).max;
        $.minTxWithdrawAmount = 0;

        // Setup role admins - all roles use ROLE_ADMIN
        ACL_OZ_5_2_0_Lib._setRoleAdmin(RolesLib.VAULT_MANAGER, RolesLib.ROLE_ADMIN);
        ACL_OZ_5_2_0_Lib._setRoleAdmin(RolesLib.HOOK_MANAGER, RolesLib.ROLE_ADMIN);
        ACL_OZ_5_2_0_Lib._setRoleAdmin(RolesLib.STRATEGY_MANAGER, RolesLib.ROLE_ADMIN);
        ACL_OZ_5_2_0_Lib._setRoleAdmin(RolesLib.ALLOCATOR, RolesLib.ROLE_ADMIN);
        ACL_OZ_5_2_0_Lib._setRoleAdmin(RolesLib.PAUSER, RolesLib.ROLE_ADMIN);

        // ROLE_ADMIN is self-admin
        ACL_OZ_5_2_0_Lib._setRoleAdmin(RolesLib.ROLE_ADMIN, RolesLib.ROLE_ADMIN);

        // Grant ROLE_ADMIN to the initial vault manager
        ACL_OZ_5_2_0_Lib._grantRole(RolesLib.ROLE_ADMIN, initialVaultManager, sender);

        // Grant manager role to the initial vault manager
        ACL_OZ_5_2_0_Lib._grantRole(RolesLib.VAULT_MANAGER, initialVaultManager, sender);
    }

    function stateInitAsyncVaultImpl(address initialVaultManager, address sender) public {
        _initAsyncCoreState();
    }

    /**
     * @dev Upgrade-path helper used by `ConcreteAsyncVaultImpl._upgrade`.
     *      Performs legacy role migration unconditionally, then runs the async-specific
     *      state init iff the async storage has never been initialised
     *      (`latestEpochID == 0` is the canonical sentinel: the init sets it to 1 and
     *      it only ever increases via `closeEpoch`).
     *      Designed to be invoked via DELEGATECALL from the vault proxy; `msg.sender` is
     *      preserved across the chain and forwarded to `migrateLegacyAdminRoles`.
     */
    function stateUpgradeAsyncVaultImpl() public {
        RoleMigrationLib.migrateLegacyAdminRoles(msg.sender);

        if (AVLib.fetch().latestEpochID == 0) _initAsyncCoreState();
    }

    /**
     * @dev Single source of truth for the async-specific core state setup.
     *      Sets `isQueueActive = true`, `latestEpochID = 1`, the default
     *      `unwindCostCapBP = 500` (5 %), emits `WithdrawalQueueInitialized`,
     *      and wires the role admins for `WITHDRAWAL_MANAGER` and
     *      `PRIORITY_WITHDRAWAL_EXECUTOR` to `ROLE_ADMIN`.
     *      Callers are responsible for any idempotency guard.
     */
    function _initAsyncCoreState() private {
        AVLib.ConcreteAsyncVaultImplStorage storage $ = AVLib.fetch();

        $.isQueueActive = true;
        $.latestEpochID = 1;
        $.unwindCostCapBP = 500;
        emit IConcreteAsyncVaultImpl.WithdrawalQueueInitialized(1);

        ACL_OZ_5_2_0_Lib._setRoleAdmin(RolesLib.WITHDRAWAL_MANAGER, RolesLib.ROLE_ADMIN);
        ACL_OZ_5_2_0_Lib._setRoleAdmin(RolesLib.PRIORITY_WITHDRAWAL_EXECUTOR, RolesLib.ROLE_ADMIN);
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/StateSetterLib.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ConcreteStandardVaultImplStorageLib as SVLib} from "./storage/ConcreteStandardVaultImplStorageLib.sol";
import {ConcreteAsyncVaultImplStorageLib as AVLib} from "./storage/ConcreteAsyncVaultImplStorageLib.sol";
import {EnumerableSet} from "@openzeppelin-contracts/utils/structs/EnumerableSet.sol";
import {IConcreteStandardVaultImpl} from "../interface/IConcreteStandardVaultImpl.sol";
import {IConcreteAsyncVaultImpl} from "../interface/IConcreteAsyncVaultImpl.sol";
import {IHurdleRateOracle} from "../interface/IHurdleRateOracle.sol";
import {ConcreteV2FeeParamsLib} from "./Constants.sol";
import {AsyncVaultHelperLib} from "./AsyncVaultHelperLib.sol";

library StateSetterLib {
    using EnumerableSet for EnumerableSet.AddressSet;

    function updateManagementFee(uint16 managementFee_) public {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        if (managementFee_ > 0) {
            require($.managementFeeRecipient != address(0), IConcreteStandardVaultImpl.FeeRecipientNotSet());
            require(
                managementFee_ <= ConcreteV2FeeParamsLib.MAX_MANAGEMENT_FEE,
                IConcreteStandardVaultImpl.ManagementFeeExceedsMaximum()
            );
        }

        $.managementFee = managementFee_;
        emit IConcreteStandardVaultImpl.ManagementFeeUpdated(managementFee_);
    }

    /// @dev Updates the management fee recipient
    /// @param recipient The new management fee recipient
    function updateManagementFeeRecipient(address recipient) external {
        require(recipient != address(0), IConcreteStandardVaultImpl.InvalidFeeRecipient());

        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        $.managementFeeRecipient = recipient;
        emit IConcreteStandardVaultImpl.ManagementFeeRecipientUpdated(recipient);
    }

    function updatePerformanceFee(uint16 performanceFee_) public {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        if (performanceFee_ > 0) {
            require($.performanceFeeRecipient != address(0), IConcreteStandardVaultImpl.FeeRecipientNotSet());
            require(
                performanceFee_ <= ConcreteV2FeeParamsLib.MAX_PERFORMANCE_FEE,
                IConcreteStandardVaultImpl.PerformanceFeeExceedsMaximum()
            );
        }

        $.performanceFee = performanceFee_;
        emit IConcreteStandardVaultImpl.PerformanceFeeUpdated(performanceFee_);
    }

    function updatePerformanceFeeRecipient(address recipient) external {
        require(recipient != address(0), IConcreteStandardVaultImpl.InvalidFeeRecipient());

        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();
        $.performanceFeeRecipient = recipient;
        emit IConcreteStandardVaultImpl.PerformanceFeeRecipientUpdated(recipient);
    }

    /**
     * @dev Sets the deposit limits for the vault.
     * @param minTxDepositAmount The minimum deposit amount per transaction.
     * @param maxGlobalDepositAmount The maximum global deposit amount (applies to total assets across all users).
     */
    function setDepositLimits(uint256 minTxDepositAmount, uint256 maxGlobalDepositAmount) public {
        require(maxGlobalDepositAmount >= minTxDepositAmount, IConcreteStandardVaultImpl.InvalidDepositLimits());

        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        $.maxGlobalDepositAmount = maxGlobalDepositAmount;
        $.minTxDepositAmount = minTxDepositAmount;
        emit IConcreteStandardVaultImpl.DepositLimitsUpdated(maxGlobalDepositAmount, minTxDepositAmount);
    }

    function setWithdrawLimits(uint256 minTxWithdrawAmount, uint256 maxTxWithdrawAmount) public {
        require(maxTxWithdrawAmount >= minTxWithdrawAmount, IConcreteStandardVaultImpl.InvalidWithdrawLimits());

        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        $.maxTxWithdrawAmount = maxTxWithdrawAmount;
        $.minTxWithdrawAmount = minTxWithdrawAmount;
        emit IConcreteStandardVaultImpl.WithdrawLimitsUpdated(maxTxWithdrawAmount, minTxWithdrawAmount);
    }

    function configure(IConcreteStandardVaultImpl.VaultManagerConfig configType, bytes calldata data) external {
        if (configType == IConcreteStandardVaultImpl.VaultManagerConfig.ManagementFee) {
            updateManagementFee(abi.decode(data, (uint16)));
        } else if (configType == IConcreteStandardVaultImpl.VaultManagerConfig.PerformanceFee) {
            updatePerformanceFee(abi.decode(data, (uint16)));
        } else if (configType == IConcreteStandardVaultImpl.VaultManagerConfig.HurdleRateOracle) {
            setHurdleRateOracle(abi.decode(data, (IHurdleRateOracle)));
        } else if (configType == IConcreteStandardVaultImpl.VaultManagerConfig.DepositLimits) {
            (uint256 minTx, uint256 maxGlobal) = abi.decode(data, (uint256, uint256));
            setDepositLimits(minTx, maxGlobal);
        } else if (configType == IConcreteStandardVaultImpl.VaultManagerConfig.WithdrawLimits) {
            (uint256 minTx, uint256 maxTx) = abi.decode(data, (uint256, uint256));
            setWithdrawLimits(minTx, maxTx);
        } else {
            revert IConcreteStandardVaultImpl.InvalidVaultManagerConfig();
        }
    }

    function setHurdleRateOracle(IHurdleRateOracle oracle) public {
        SVLib.fetch().hurdleRateOracle = oracle;
        emit IConcreteStandardVaultImpl.HurdleRateOracleUpdated(address(oracle));
    }

    function addStrategy(address strategy) external {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        require($.strategies.add(strategy), IConcreteStandardVaultImpl.StrategyAlreadyAdded());
        $.strategyData[strategy] = IConcreteStandardVaultImpl.StrategyData({
            status: IConcreteStandardVaultImpl.StrategyStatus.Active, allocated: 0
        });

        emit IConcreteStandardVaultImpl.StrategyAdded(strategy);
    }

    function removeStrategy(address strategy) external {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        IConcreteStandardVaultImpl.StrategyData memory strategyDataCached = $.strategyData[strategy];

        require(
            (strategyDataCached.allocated == 0 && _strategyNotInDeallocationOrder(strategy))
                || strategyDataCached.status == IConcreteStandardVaultImpl.StrategyStatus.Halted,
            IConcreteStandardVaultImpl.StrategyHasAllocation()
        );
        require($.strategies.remove(strategy), IConcreteStandardVaultImpl.StrategyDoesNotExist());

        delete $.strategyData[strategy];
        _removeStrategyFromDeallocationOrder(strategy);

        emit IConcreteStandardVaultImpl.StrategyRemoved(strategy);
    }

    /**
     * @dev Removes all instances of a strategy from the deallocation order array.
     *      Preserves the relative order of remaining strategies.
     *      Handles duplicated entries (e.g. from setDeallocationOrder with duplicates).
     * @param strategy The strategy address to remove from deallocation order.
     */
    function _removeStrategyFromDeallocationOrder(address strategy) internal {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        address[] storage deallocationOrder = $.deallocationOrder;
        uint256 i;
        while (i < deallocationOrder.length) {
            if (deallocationOrder[i] == strategy) {
                // Shift elements to preserve relative order of remaining strategies
                for (uint256 j = i; j < deallocationOrder.length - 1; ++j) {
                    deallocationOrder[j] = deallocationOrder[j + 1];
                }
                deallocationOrder.pop();
                emit IConcreteStandardVaultImpl.DeallocationOrderUpdated();
                // Do not increment i - next element is now at position i
            } else {
                ++i;
            }
        }
    }

    function toggleStrategyStatus(address strategy) external {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        require($.strategies.contains(strategy), IConcreteStandardVaultImpl.StrategyDoesNotExist());

        IConcreteStandardVaultImpl.StrategyStatus currentStatus = $.strategyData[strategy].status;

        if (currentStatus == IConcreteStandardVaultImpl.StrategyStatus.Active) {
            $.strategyData[strategy].status = IConcreteStandardVaultImpl.StrategyStatus.Halted;
        } else {
            $.strategyData[strategy].status = IConcreteStandardVaultImpl.StrategyStatus.Active;
        }

        emit IConcreteStandardVaultImpl.StrategyStatusToggled(strategy);
    }

    /**
     * @dev Internal function to check if a strategy is not in the deallocation order.
     * @param strategy The strategy address to check.
     * @return True if the strategy is not in the deallocation order, false otherwise.
     */
    function _strategyNotInDeallocationOrder(address strategy) internal view returns (bool) {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        address[] memory deallocationOrder = $.deallocationOrder;
        uint256 deallocationOrderLength = deallocationOrder.length;
        for (uint256 i = 0; i < deallocationOrderLength; i++) {
            if (deallocationOrder[i] == strategy) {
                return false;
            }
        }
        return true;
    }

    /**
     * @notice overwrites the deallocation order from strategies;
     */
    function setDeallocationOrder(address[] calldata order) external {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        delete $.deallocationOrder;

        uint256 orderLength = order.length;
        for (uint256 i = 0; i < orderLength; i++) {
            address strategy = order[i];
            require($.strategies.contains(strategy), IConcreteStandardVaultImpl.StrategyDoesNotExist());
            require(
                $.strategyData[strategy].status == IConcreteStandardVaultImpl.StrategyStatus.Active,
                IConcreteStandardVaultImpl.StrategyIsHalted()
            );
        }

        $.deallocationOrder = order;

        emit IConcreteStandardVaultImpl.DeallocationOrderUpdated();
    }

    /**
     * @notice Set the unwind cost cap for priority withdrawals
     * @dev Validates that the unwind cost cap does not exceed BASIS_POINTS
     * @param unwindCostCap_ The new unwind cost cap in basis points (10000 = 100%)
     */
    function setUnwindCostCap(uint16 unwindCostCap_) external {
        require(unwindCostCap_ <= AsyncVaultHelperLib.BASIS_POINTS, IConcreteAsyncVaultImpl.InvalidUnwindCostCap());
        AVLib.ConcreteAsyncVaultImplStorage storage $ = AVLib.fetch();
        $.unwindCostCapBP = unwindCostCap_;
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/ContractIdEncoding.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title ContractIdEncoding
 * @notice Library for encoding and decoding contract IDs with version, flavour, and contract type
 * @dev The ID structure is as follows:
 *      - Version: 0-99999 (5 digits)
 *          - Major release: 0-9 (first digit)
 *          - Minor release: 0-99 (next 2 digits)
 *          - Patch/Batch: 0-99 (last 2 digits)
 *          - Version 99999 is reserved for tests or otherwise non-production versions.
 *      - Flavours: 100000-99900000 (next 3 digits, multiplied by 100000)
 *          - E.g. for vault contract types we could have:
 *             0: Concrete Standard Vault
 *             1: Concrete Async Vault
 *             2: Concrete Standard Predeposit Vault
 *             3: Concrete Async Predeposit Vault
 *             4: Concrete Standard Bridged Vault
 *             5: Concrete Async Bridged Vault
 *          - flavour 999 is reserved for tests or otherwise non-production flavours.
 *      - Contract types: 100000000-9900000000 (next 2 digits, multiplied by 100000000)
 *
 *      Contract type mappings:
 *      - 0: Vault contract type
 *      - 1: Multi Asset Vault contract type
 *      - 2: Strategy contract type
 *      - 3: Hook contract type
 *      - 4: Fee splitter contract type
 */
library ContractIdEncoding {
    /// @notice Maximum major version value (9)
    uint256 public constant MAX_MAJOR_VERSION = 9;

    /// @notice Maximum minor version value (99)
    uint256 public constant MAX_MINOR_VERSION = 99;

    /// @notice Maximum patch version value (99)
    uint256 public constant MAX_PATCH_VERSION = 99;

    /// @notice Maximum version value (99999)
    uint256 public constant MAX_VERSION = 99999;

    /// @notice Maximum flavour value (999)
    uint256 public constant MAX_FLAVOUR = 999;

    /// @notice Maximum contract type value (99)
    uint256 public constant MAX_CONTRACT_TYPE = 99;

    /// @notice Multiplier for flavour position
    uint256 private constant FLAVOUR_MULTIPLIER = 100000;

    /// @notice Multiplier for contract type position
    uint256 private constant CONTRACT_TYPE_MULTIPLIER = 100000000;

    /**
     * @notice Encodes contract type, flavour, and version into a single ID
     * @param contractType The contract type (0-99)
     *                  - 0: Vault contract type
     *                  - 1: Strategy contract type
     *                  - 2: Hook contract type
     *                  - 3: Fee splitter contract type
     * @param flavour The flavour (0-999)
     * @param majorVersion The major version (0-9)
     * @param minorVersion The minor version (0-99)
     * @param patchVersion The patch/batch version (0-99)
     * @return encodedId The encoded ID (uint64)
     */
    function encode(
        uint256 contractType,
        uint256 flavour,
        uint256 majorVersion,
        uint256 minorVersion,
        uint256 patchVersion
    ) internal pure returns (uint64 encodedId) {
        require(contractType <= MAX_CONTRACT_TYPE, "IdEncoding: contractType exceeds max");
        require(flavour <= MAX_FLAVOUR, "IdEncoding: flavour exceeds max");
        require(majorVersion <= MAX_MAJOR_VERSION, "IdEncoding: majorVersion exceeds max");
        require(minorVersion <= MAX_MINOR_VERSION, "IdEncoding: minorVersion exceeds max");
        require(patchVersion <= MAX_PATCH_VERSION, "IdEncoding: patchVersion exceeds max");

        uint256 version = majorVersion * 10000 + minorVersion * 100 + patchVersion;
        encodedId = uint64(version + (flavour * FLAVOUR_MULTIPLIER) + (contractType * CONTRACT_TYPE_MULTIPLIER));
    }

    /**
     * @notice Decodes an encoded ID into contract type, flavour, and version components
     * @param encodedId The encoded ID to decode (uint64)
     * @return contractType The contract type (0-99)
     *                  - 0: Single Asset Vault contract type
     *                  - 1: Multi Asset Vault contract type
     *                  - 2: Strategy contract type
     *                  - 3: Hook contract type
     *                  - 4: Fee splitter contract type
     * @return flavour The flavour (0-999)
     * @return majorVersion The major version (0-9)
     * @return minorVersion The minor version (0-99)
     * @return patchVersion The patch/batch version (0-99)
     */
    function decode(uint64 encodedId)
        internal
        pure
        returns (
            uint256 contractType,
            uint256 flavour,
            uint256 majorVersion,
            uint256 minorVersion,
            uint256 patchVersion
        )
    {
        uint256 version = uint256(encodedId) % FLAVOUR_MULTIPLIER;
        flavour = (uint256(encodedId) / FLAVOUR_MULTIPLIER) % (CONTRACT_TYPE_MULTIPLIER / FLAVOUR_MULTIPLIER);
        contractType = uint256(encodedId) / CONTRACT_TYPE_MULTIPLIER;

        majorVersion = version / 10000;
        minorVersion = (version / 100) % 100;
        patchVersion = version % 100;
    }
}



// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/WithdrawLib.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ConcreteStandardVaultImplStorageLib as SVLib} from "./storage/ConcreteStandardVaultImplStorageLib.sol";
import {IConcreteStandardVaultImpl} from "../interface/IConcreteStandardVaultImpl.sol";
import {IStrategyTemplate} from "../interface/IStrategyTemplate.sol";
import {IERC20} from "@openzeppelin-contracts/token/ERC20/IERC20.sol";
import {SafeCast} from "@openzeppelin-contracts/utils/math/SafeCast.sol";
import {EnumerableSet} from "@openzeppelin-contracts/utils/structs/EnumerableSet.sol";

/**
 * @title WithdrawLib
 * @notice Library for handling withdrawal operations from vault strategies.
 * @dev Contains public functions to reduce bytecode size of the main vault contract
 *      via delegatecall pattern. All functions access vault storage through EIP-7201
 *      namespaced storage.
 */
library WithdrawLib {
    using SafeCast for uint256;
    using EnumerableSet for EnumerableSet.AddressSet;

    /**
     * @dev Executes withdrawals from strategies to fulfill a withdrawal request.
     * @dev This function iterates through strategies in deallocation order and withdraws
     *      assets until the requested amount is reached or all strategies are exhausted.
     * @param assetAddress The address of the underlying asset
     * @param requestedAssets The amount of assets requested to withdraw
     * @param lockedAssets The amount of assets that are locked and not available for withdrawal
     * @return totalWithdrawableAmount The total amount of assets that can be withdrawn
     */
    function executeWithdrawFromStrategies(address assetAddress, uint256 requestedAssets, uint256 lockedAssets)
        public
        returns (uint256 totalWithdrawableAmount)
    {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        uint256 floatingFunds = IERC20(assetAddress).balanceOf(address(this));
        totalWithdrawableAmount = floatingFunds >= lockedAssets ? floatingFunds - lockedAssets : 0;

        if (totalWithdrawableAmount < requestedAssets) {
            // Iterate directly over storage to avoid copying the entire array to memory
            // This is especially beneficial for small withdrawals that only need the first strategy
            uint256 strategiesCounter = $.deallocationOrder.length;

            for (uint256 i; i < strategiesCounter; ++i) {
                address strategy = $.deallocationOrder[i];
                require($.strategies.contains(strategy), IConcreteStandardVaultImpl.StaleDeallocationOrder(strategy));
                if ($.strategyData[strategy].status != IConcreteStandardVaultImpl.StrategyStatus.Active) continue;

                uint256 desiredAssets;
                unchecked {
                    desiredAssets = requestedAssets - totalWithdrawableAmount;
                }

                uint256 withdrawableAmountFromStrategy = IStrategyTemplate(strategy).maxWithdraw();
                uint256 withdrawAmount =
                    (withdrawableAmountFromStrategy >= desiredAssets) ? desiredAssets : withdrawableAmountFromStrategy;

                if (withdrawAmount > 0) {
                    uint256 actualWithdrawn = IStrategyTemplate(strategy).onWithdraw(withdrawAmount);
                    $.strategyData[strategy].allocated -= actualWithdrawn.toUint120();

                    totalWithdrawableAmount += actualWithdrawn;
                }

                if (totalWithdrawableAmount >= requestedAssets) break;
            }
        }
    }

    /**
     * @dev Simulates withdrawing an amount of assets from the vault without modifying state.
     * @dev This is a view function that calculates how much can be withdrawn by iterating
     *      through strategies, but does NOT update strategy allocated amounts.
     * @param assetAddress The address of the underlying asset
     * @param requestedAssets Amount of assets to withdraw
     * @param lockedAssets The amount of assets that are locked and not available for withdrawal
     * @return totalWithdrawableAmount Amount of assets that can be filled
     */
    function simulateWithdraw(address assetAddress, uint256 requestedAssets, uint256 lockedAssets)
        public
        view
        returns (uint256 totalWithdrawableAmount)
    {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        uint256 floatingFunds = IERC20(assetAddress).balanceOf(address(this));
        totalWithdrawableAmount = floatingFunds >= lockedAssets ? floatingFunds - lockedAssets : 0;

        if (totalWithdrawableAmount < requestedAssets) {
            // Iterate directly over storage to avoid copying the entire array to memory
            // This is especially beneficial for small withdrawals that only need the first strategy
            uint256 strategiesCounter = $.deallocationOrder.length;

            for (uint256 i; i < strategiesCounter; ++i) {
                address strategy = $.deallocationOrder[i];
                require($.strategies.contains(strategy), IConcreteStandardVaultImpl.StaleDeallocationOrder(strategy));
                if ($.strategyData[strategy].status != IConcreteStandardVaultImpl.StrategyStatus.Active) continue;

                uint256 desiredAssets;
                unchecked {
                    desiredAssets = requestedAssets - totalWithdrawableAmount;
                }

                uint256 withdrawableAmountFromStrategy = IStrategyTemplate(strategy).maxWithdraw();
                uint256 withdrawAmount =
                    (withdrawableAmountFromStrategy >= desiredAssets) ? desiredAssets : withdrawableAmountFromStrategy;

                totalWithdrawableAmount += withdrawAmount;

                if (totalWithdrawableAmount >= requestedAssets) break;
            }
        } else {
            totalWithdrawableAmount = requestedAssets;
        }

        // ERC4626 compliance: if available liquidity falls below minTxWithdrawAmount,
        // return 0 so maxWithdraw/maxRedeem never advertise an amount that withdraw/redeem would reject.
        if (totalWithdrawableAmount < $.minTxWithdrawAmount) {
            totalWithdrawableAmount = 0;
        }
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/YieldAccrualLib.sol =====
// SPDX-License-Identifier: AGPL-3.0
pragma solidity ^0.8.24;

import {Math} from "@openzeppelin-contracts/utils/math/Math.sol";
import {SafeCast} from "@openzeppelin-contracts/utils/math/SafeCast.sol";
import {EnumerableSet} from "@openzeppelin-contracts/utils/structs/EnumerableSet.sol";

import {ConcreteStandardVaultImplStorageLib as SVLib} from "./storage/ConcreteStandardVaultImplStorageLib.sol";
import {
    ConcreteCachedVaultStateStorageLib as CachedVaultStateLib
} from "./storage/ConcreteCachedVaultStateStorageLib.sol";

import {ConcreteV2ConstantsLib as ConstantsLib} from "./Constants.sol";
import {ConcreteV2ConversionLib as ConversionLib} from "./Conversion.sol";
import {ERC20_OZ_5_2_0_Lib} from "./ERC20Lib.sol";
import {Time} from "./Time.sol";

import {IConcreteStandardVaultImpl} from "../interface/IConcreteStandardVaultImpl.sol";
import {IHurdleRateOracle} from "../interface/IHurdleRateOracle.sol";
import {IStrategyTemplate} from "../interface/IStrategyTemplate.sol";

library YieldAccrualLib {
    using EnumerableSet for EnumerableSet.AddressSet;
    using SafeCast for uint256;

    /**
     * @dev Executes the yield accrual operation across all active strategies.
     * @return totalAssetsCached The updated total assets after yield accrual
     */
    function accrueYield() public returns (uint256) {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        uint256 totalPositiveYield;
        uint256 totalNegativeYield;
        {
            address[] memory strategies = $.strategies.values();

            for (uint256 i; i < strategies.length; ++i) {
                (uint256 positiveYield, uint256 loss, uint256 strategyTotalAllocatedValue) =
                    previewStrategyYield(strategies[i]);

                if (positiveYield != 0 || loss != 0) {
                    $.strategyData[strategies[i]].allocated = strategyTotalAllocatedValue.toUint120();

                    totalPositiveYield += positiveYield;
                    totalNegativeYield += loss;

                    emit IConcreteStandardVaultImpl.StrategyYieldAccrued(
                        strategies[i], strategyTotalAllocatedValue, positiveYield, loss
                    );
                }
            }
        }

        CachedVaultStateLib.ConcreteCachedVaultStateStorage storage $cached = CachedVaultStateLib.fetch();

        uint256 totalAssetsCached = $cached.cachedTotalAssets + totalPositiveYield - totalNegativeYield;
        $cached.cachedTotalAssets = totalAssetsCached;

        // Accrue management fees after accruing yield
        accrueManagementFee(totalAssetsCached);

        // Accrue performance fees on net yield amount
        accruePerformanceFee(totalAssetsCached, totalPositiveYield, totalNegativeYield);

        emit IConcreteStandardVaultImpl.YieldAccrued(totalPositiveYield, totalNegativeYield);

        return totalAssetsCached;
    }

    /**
     * @dev Accrue management fees by minting shares to the fee recipient.
     * @param totalAssetsAmount The total assets in the vault to calculate fee on
     */
    function accrueManagementFee(uint256 totalAssetsAmount) public {
        (uint256 feeShares, uint256 feeAmount) = previewManagementFee(totalAssetsAmount);

        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        // Update last accrual timestamp
        $.lastManagementFeeAccrual = Time.timestamp();

        // Mint shares to management fee recipient
        address managementFeeRecipient = $.managementFeeRecipient;
        if (feeShares != 0 && managementFeeRecipient != address(0)) {
            ERC20_OZ_5_2_0_Lib._mint(managementFeeRecipient, feeShares);

            emit IConcreteStandardVaultImpl.ManagementFeeAccrued(managementFeeRecipient, feeShares, feeAmount);
        }
    }

    /**
     * @dev Accrue performance fees by minting shares to the fee recipient.
     * @param totalAssetsAmount The total assets in the vault to calculate fee on
     * @param positiveYield The total positive yield generated by all strategies
     * @param loss The total losses incurred by all strategies
     */
    function accruePerformanceFee(uint256 totalAssetsAmount, uint256 positiveYield, uint256 loss) public {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        if ($.performanceFee == 0 || (loss >= positiveYield)) return;

        uint256 totalSupply_ = ERC20_OZ_5_2_0_Lib.totalSupply();

        (uint256 hurdleRate, uint256 oraclePrecision) = _fetchHurdleRate($.hurdleRateOracle);

        (uint256 performanceFeeShares, uint256 feeAmount) = _previewPerformanceFee(
            totalAssetsAmount, positiveYield, loss, totalSupply_, $.performanceFee, hurdleRate, oraclePrecision
        );

        if (performanceFeeShares == 0) return;

        address performanceFeeRecipient = $.performanceFeeRecipient;
        if (performanceFeeRecipient != address(0)) {
            ERC20_OZ_5_2_0_Lib._mint(performanceFeeRecipient, performanceFeeShares);
            emit IConcreteStandardVaultImpl.PerformanceFeeAccrued(
                performanceFeeRecipient, performanceFeeShares, feeAmount
            );
        }
    }

    /**
     * @dev Simulates the yield accrual operation across all strategies including fees.
     * @param cachedTotalAssets The current cached total assets
     * @return totalAssets_ The projected total assets after yield accrual
     * @return totalSupply_ The projected total supply after yield accrual
     */
    function previewAccrueYieldAndFees(uint256 cachedTotalAssets) public view returns (uint256, uint256) {
        (uint256 totalPositiveYield, uint256 totalNegativeYield) = previewYieldNoFees();

        uint256 totalSupplyCached = ERC20_OZ_5_2_0_Lib.totalSupply();
        uint256 totalAssetsCached = cachedTotalAssets + totalPositiveYield - totalNegativeYield;

        (uint256 managementFeeShares,) = previewManagementFee(totalAssetsCached);
        totalSupplyCached += managementFeeShares;

        (uint256 performanceFeeShares,) =
            previewPerformanceFee(totalAssetsCached, totalPositiveYield, totalNegativeYield, totalSupplyCached);
        totalSupplyCached += performanceFeeShares;

        return (totalAssetsCached, totalSupplyCached);
    }

    /**
     * @dev Calculates total positive and negative yield across all strategies.
     * @return totalPositiveYield The sum of all positive yields from strategies
     * @return totalNegativeYield The sum of all losses from strategies
     */
    function previewYieldNoFees() public view returns (uint256 totalPositiveYield, uint256 totalNegativeYield) {
        address[] memory strategies = SVLib.fetch().strategies.values();

        for (uint256 i; i < strategies.length; ++i) {
            (uint256 positiveYield, uint256 loss,) = previewStrategyYield(strategies[i]);

            if (positiveYield != 0 || loss != 0) {
                totalPositiveYield += positiveYield;
                totalNegativeYield += loss;
            }
        }
    }

    /**
     * @dev Preview yield for a single strategy.
     * @param strategy The address of the strategy contract
     * @return yield The amount of positive yield generated
     * @return loss The amount of loss incurred
     * @return currentTotalAllocatedValue The current total allocated value
     */
    function previewStrategyYield(address strategy) public view returns (uint256, uint256, uint256) {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        uint120 strategyAllocatedAmount = $.strategyData[strategy].allocated;

        if ($.strategyData[strategy].status != IConcreteStandardVaultImpl.StrategyStatus.Active) {
            return (0, 0, 0);
        }

        uint256 currentTotalAllocatedValue = IStrategyTemplate(strategy).totalAllocatedValue();
        currentTotalAllocatedValue =
            (currentTotalAllocatedValue >= type(uint120).max) ? type(uint120).max : currentTotalAllocatedValue;

        uint256 yield;
        uint256 loss;
        if (currentTotalAllocatedValue == strategyAllocatedAmount) {
            return (yield, loss, currentTotalAllocatedValue);
        } else if (currentTotalAllocatedValue > strategyAllocatedAmount) {
            yield = currentTotalAllocatedValue - strategyAllocatedAmount;
        } else {
            loss = strategyAllocatedAmount - currentTotalAllocatedValue;
        }

        return (yield, loss, currentTotalAllocatedValue);
    }

    /**
     * @dev Preview management fee accrual.
     * @param _lastTotalAssets The total assets to calculate fee on
     * @return feeShares The number of shares to mint as management fee
     * @return feeAmount The asset value of the management fee
     */
    function previewManagementFee(uint256 _lastTotalAssets) public view returns (uint256 feeShares, uint256 feeAmount) {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        if ($.managementFee == 0) return (0, 0);

        uint32 currentTime = Time.timestamp();
        uint32 lastAccrual = $.lastManagementFeeAccrual;

        if (currentTime == lastAccrual) return (0, 0);

        uint256 timeElapsed = currentTime - lastAccrual;
        uint256 annualFeeAmount = (_lastTotalAssets * $.managementFee) / ConstantsLib.BASIS_POINTS_DENOMINATOR;
        feeAmount = (annualFeeAmount * timeElapsed) / (365 days);

        if (feeAmount == 0) return (0, 0);

        // sanity check - clamp the fee amount to the last total assets
        if (feeAmount > _lastTotalAssets) {
            feeAmount = _lastTotalAssets;
        }

        // convert fee amount to shares using total assets net of the fee
        feeShares = ConversionLib.calcConvertToShares(
            feeAmount, ERC20_OZ_5_2_0_Lib.totalSupply(), _lastTotalAssets - feeAmount, Math.Rounding.Floor, false
        );

        return (feeShares, feeAmount);
    }

    /**
     * @dev Preview performance fee accrual.
     * @param totalAssetsAmount The total assets in the vault (post-yield, pre-fee)
     * @param positiveYield The total positive yield generated
     * @param loss The total losses incurred
     * @param totalSupply_ The current total supply of vault shares
     * @return shares The number of shares to mint as performance fee
     * @return feeAmount The asset value of the performance fee
     */
    function previewPerformanceFee(uint256 totalAssetsAmount, uint256 positiveYield, uint256 loss, uint256 totalSupply_)
        public
        view
        returns (uint256, uint256)
    {
        SVLib.ConcreteStandardVaultImplStorage storage $ = SVLib.fetch();

        if ($.performanceFee == 0 || (loss >= positiveYield)) return (0, 0);

        (uint256 hurdleRate, uint256 oraclePrecision) = _fetchHurdleRate($.hurdleRateOracle);

        return _previewPerformanceFee(
            totalAssetsAmount, positiveYield, loss, totalSupply_, $.performanceFee, hurdleRate, oraclePrecision
        );
    }

    /**
     * @dev Shared performance fee computation used by both the view preview
     *      and the state-mutating accrual paths.
     */
    function _previewPerformanceFee(
        uint256 totalAssetsAmount,
        uint256 positiveYield,
        uint256 loss,
        uint256 totalSupply_,
        uint16 performanceFee,
        uint256 hurdleRate,
        uint256 oraclePrecision
    ) private pure returns (uint256 feeShares, uint256 feeAmount) {
        uint256 netPositiveYield = positiveYield - loss;

        if (hurdleRate > 0) {
            netPositiveYield = _capNetYieldToExcessAboveHurdleRate(
                totalAssetsAmount, netPositiveYield, hurdleRate, totalSupply_, oraclePrecision
            );
        }

        feeAmount = Math.mulDiv(netPositiveYield, performanceFee, ConstantsLib.BASIS_POINTS_DENOMINATOR);

        if (feeAmount == 0) return (0, 0);

        feeShares = ConversionLib.calcConvertToShares(
            feeAmount, totalSupply_, totalAssetsAmount - feeAmount, Math.Rounding.Floor, false
        );
    }

    /**
     * @dev Reads the hurdle rate from the oracle.
     *      Returns (0, 0) when no oracle is configured.
     */
    function _fetchHurdleRate(IHurdleRateOracle oracle)
        private
        view
        returns (uint256 hurdleRate, uint256 oraclePrecision)
    {
        if (address(oracle) == address(0)) return (0, 0);
        hurdleRate = oracle.getHurdleRate();
        oraclePrecision = oracle.precision();
    }

    /**
     * @dev Adjust the total assets and total supply by 1 virtual share consistent with share price calculation.
     * @dev hurdleAssets uses simple truncation rounding in favour of fee recipient
     */
    function _capNetYieldToExcessAboveHurdleRate(
        uint256 totalAssetsAmount,
        uint256 netPositiveYield,
        uint256 hurdleRate,
        uint256 totalSupply_,
        uint256 oraclePrecision
    ) private pure returns (uint256 yieldAboveHurdle) {
        uint256 hurdleAssets = Math.mulDiv(hurdleRate, totalSupply_ + 1, oraclePrecision);
        uint256 excessAboveHurdle = (totalAssetsAmount + 1) <= hurdleAssets ? 0 : (totalAssetsAmount + 1) - hurdleAssets;
        yieldAboveHurdle = netPositiveYield > excessAboveHurdle ? excessAboveHurdle : netPositiveYield;
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/storage/ConcreteCachedVaultStateStorageLib.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/types/Time.sol)
pragma solidity ^0.8.24;

library ConcreteCachedVaultStateStorageLib {
    /// @custom:storage-location erc7201:openzeppelin.storage.ConcreteTokenizedVaultStorage
    struct ConcreteCachedVaultStateStorage {
        /// @dev Store last total assets of the vault at specific timestamp
        uint256 cachedTotalAssets;
    }

    /// @dev keccak256(abi.encode(uint256(keccak256("concrete.storage.ConcreteCachedVaultStateStorage")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant ConcreteCachedVaultStateStorageLocation =
        0x31b60059595cae2ebab32b53f301cf68fb9c4eef322a90dbc8487ddf3a197900;

    function fetch() internal pure returns (ConcreteCachedVaultStateStorage storage $) {
        assembly {
            $.slot := ConcreteCachedVaultStateStorageLocation
        }
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/storage/ConcreteStandardVaultImplStorageLib.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/types/Time.sol)
pragma solidity ^0.8.24;

import {IConcreteStandardVaultImpl} from "../../interface/IConcreteStandardVaultImpl.sol";
import {IHurdleRateOracle} from "../../interface/IHurdleRateOracle.sol";
import {EnumerableSet} from "@openzeppelin-contracts/utils/structs/EnumerableSet.sol";
import {Hooks} from "../Hooks.sol";

library ConcreteStandardVaultImplStorageLib {
    /// @dev keccak256(abi.encode(uint256(keccak256("concrete.storage.ConcreteStandardVaultImplStorage")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant ConcreteStandardVaultImplStorageLocation =
        0xe74d828616eceb28be4a8cf3f9eeee868e1f44ce928ee17a9d7ad296fa52be00;

    /// @custom:storage-location erc7201:concrete.storage.ConcreteStandardVaultImplStorage
    struct ConcreteStandardVaultImplStorage {
        /// @dev max global deposit amount
        uint256 maxGlobalDepositAmount;
        /// @dev max tx withdraw amount
        uint256 maxTxWithdrawAmount;
        /// @dev min tx deposit amount
        uint256 minTxDepositAmount;
        /// @dev min tx withdraw amount
        uint256 minTxWithdrawAmount;
        /// @dev allocate module's address
        address allocateModule;
        /// 1 slot: 160 + 16 + 32
        /// @dev management fee recipient
        address managementFeeRecipient;
        /// @dev annual management fee rate in basis points
        uint16 managementFee;
        /// @dev timestamp of last management fee accrual
        uint32 lastManagementFeeAccrual;
        /// 1 slot: 160 + 16
        /// @dev performance fee recipient
        address performanceFeeRecipient;
        /// @dev annual performance fee rate in basis points
        uint16 performanceFee;
        /// @dev high water mark
        uint128 performanceFeeHighWaterMark;
        /// Mapping between a strategy address and it's data
        mapping(address => IConcreteStandardVaultImpl.StrategyData) strategyData;
        /// An set of strategy addresses that ConcreteVault allocates to
        EnumerableSet.AddressSet strategies;
        /// Defines the order in which funds are retrieved from strategies to fulfill withdrawals
        address[] deallocationOrder;
        /// @dev hooks
        Hooks hooks;
        /// @dev Optional hurdle rate oracle; when set (non-zero), performance fees
        ///      are only charged on yield exceeding the oracle's reported exchange rate.
        IHurdleRateOracle hurdleRateOracle;
    }

    /**
     *
     */
    function fetch() internal pure returns (ConcreteStandardVaultImplStorage storage $) {
        assembly {
            $.slot := ConcreteStandardVaultImplStorageLocation
        }
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/RoleMigrationLib.sol =====
// SPDX-License-Identifier: AGPL-3.0
pragma solidity ^0.8.24;

import {ACL_OZ_5_2_0_Lib} from "./AccessControlLib.sol";
import {ConcreteV2RolesLib as RolesLib} from "./Roles.sol";
import {LegacyAdminRolesLib} from "./LegacyAdminRolesLib.sol";

/**
 * @title RoleMigrationLib
 * @dev Library for migrating from legacy admin roles to ROLE_ADMIN
 */
library RoleMigrationLib {
    /**
     * @dev Migrates legacy admin roles to ROLE_ADMIN
     * @dev Gets the first VAULT_MANAGER_ADMIN member (if any) and grants them ROLE_ADMIN
     * @dev Updates all role admin relationships to use ROLE_ADMIN
     * @dev Then revokes all legacy admin roles from all their members
     * @param sender The address performing the migration (typically the factory or upgrade caller)
     */
    function migrateLegacyAdminRoles(address sender) external {
        // Get all legacy admin roles
        bytes32[] memory legacyRoles = LegacyAdminRolesLib.getAllLegacyAdminRoles();

        // Try to get the first VAULT_MANAGER_ADMIN member
        address firstVaultManagerAdmin = _getFirstRoleMember(LegacyAdminRolesLib.VAULT_MANAGER_ADMIN);

        // If VAULT_MANAGER_ADMIN has members, grant ROLE_ADMIN to the first one
        if (firstVaultManagerAdmin != address(0)) {
            ACL_OZ_5_2_0_Lib._grantRole(RolesLib.ROLE_ADMIN, firstVaultManagerAdmin, sender);

            // Update all role admin relationships to use ROLE_ADMIN
            // This must be done BEFORE revoking legacy admin roles, otherwise we won't have permission
            ACL_OZ_5_2_0_Lib._setRoleAdmin(RolesLib.VAULT_MANAGER, RolesLib.ROLE_ADMIN);
            ACL_OZ_5_2_0_Lib._setRoleAdmin(RolesLib.HOOK_MANAGER, RolesLib.ROLE_ADMIN);
            ACL_OZ_5_2_0_Lib._setRoleAdmin(RolesLib.STRATEGY_MANAGER, RolesLib.ROLE_ADMIN);
            ACL_OZ_5_2_0_Lib._setRoleAdmin(RolesLib.ALLOCATOR, RolesLib.ROLE_ADMIN);
            ACL_OZ_5_2_0_Lib._setRoleAdmin(RolesLib.WITHDRAWAL_MANAGER, RolesLib.ROLE_ADMIN);
            ACL_OZ_5_2_0_Lib._setRoleAdmin(RolesLib.PAUSER, RolesLib.ROLE_ADMIN);

            // ROLE_ADMIN is self-admin
            ACL_OZ_5_2_0_Lib._setRoleAdmin(RolesLib.ROLE_ADMIN, RolesLib.ROLE_ADMIN);

            // Revoke all legacy admin roles from all their members
            for (uint256 i = 0; i < legacyRoles.length; i++) {
                _revokeRoleFromAllMembers(legacyRoles[i], sender);
            }
        }
    }

    /**
     * @dev Gets the first member of a role, or address(0) if the role has no members
     * @param role The role to get the first member from
     * @return The first member address, or address(0) if none
     */
    function _getFirstRoleMember(bytes32 role) private view returns (address) {
        uint256 memberCount = ACL_OZ_5_2_0_Lib._getRoleMemberCount(role);
        if (memberCount == 0) {
            return address(0);
        }
        return ACL_OZ_5_2_0_Lib._getRoleMember(role, 0);
    }

    /**
     * @dev Revokes a role from all its members
     * @param role The role to revoke from all members
     * @param sender The address performing the revocation
     */
    function _revokeRoleFromAllMembers(bytes32 role, address sender) private {
        uint256 memberCount = ACL_OZ_5_2_0_Lib._getRoleMemberCount(role);

        // Iterate backwards to avoid index shifting issues when removing members
        // Note: EnumerableSet removes by swapping with last element, so we iterate backwards
        for (uint256 i = memberCount; i > 0; i--) {
            address member = ACL_OZ_5_2_0_Lib._getRoleMember(role, i - 1);
            ACL_OZ_5_2_0_Lib._revokeRole(role, member, sender);
        }
    }
}



// ===== FILE: node_modules/_openzeppelin/contracts/access/extensions/IAccessControlEnumerable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (access/extensions/IAccessControlEnumerable.sol)

pragma solidity ^0.8.20;

import {IAccessControl} from "../IAccessControl.sol";

/**
 * @dev External interface of AccessControlEnumerable declared to support ERC-165 detection.
 */
interface IAccessControlEnumerable is IAccessControl {
    /**
     * @dev Returns one of the accounts that have `role`. `index` must be a
     * value between 0 and {getRoleMemberCount}, non-inclusive.
     *
     * Role bearers are not sorted in any particular way, and their ordering may
     * change at any point.
     *
     * WARNING: When using {getRoleMember} and {getRoleMemberCount}, make sure
     * you perform all queries on the same block. See the following
     * https://forum.openzeppelin.com/t/iterating-over-elements-on-enumerableset-in-openzeppelin-contracts/2296[forum post]
     * for more information.
     */
    function getRoleMember(bytes32 role, uint256 index) external view returns (address);

    /**
     * @dev Returns the number of accounts that have `role`. Can be used
     * together with {getRoleMember} to enumerate all bearers of a role.
     */
    function getRoleMemberCount(bytes32 role) external view returns (uint256);
}


// ===== FILE: node_modules/_openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (access/AccessControl.sol)

pragma solidity ^0.8.20;

import {IAccessControl} from "@openzeppelin/contracts/access/IAccessControl.sol";
import {ContextUpgradeable} from "../utils/ContextUpgradeable.sol";
import {ERC165Upgradeable} from "../utils/introspection/ERC165Upgradeable.sol";
import {Initializable} from "../proxy/utils/Initializable.sol";

/**
 * @dev Contract module that allows children to implement role-based access
 * control mechanisms. This is a lightweight version that doesn't allow enumerating role
 * members except through off-chain means by accessing the contract event logs. Some
 * applications may benefit from on-chain enumerability, for those cases see
 * {AccessControlEnumerable}.
 *
 * Roles are referred to by their `bytes32` identifier. These should be exposed
 * in the external API and be unique. The best way to achieve this is by
 * using `public constant` hash digests:
 *
 * ```solidity
 * bytes32 public constant MY_ROLE = keccak256("MY_ROLE");
 * ```
 *
 * Roles can be used to represent a set of permissions. To restrict access to a
 * function call, use {hasRole}:
 *
 * ```solidity
 * function foo() public {
 *     require(hasRole(MY_ROLE, msg.sender));
 *     ...
 * }
 * ```
 *
 * Roles can be granted and revoked dynamically via the {grantRole} and
 * {revokeRole} functions. Each role has an associated admin role, and only
 * accounts that have a role's admin role can call {grantRole} and {revokeRole}.
 *
 * By default, the admin role for all roles is `DEFAULT_ADMIN_ROLE`, which means
 * that only accounts with this role will be able to grant or revoke other
 * roles. More complex role relationships can be created by using
 * {_setRoleAdmin}.
 *
 * WARNING: The `DEFAULT_ADMIN_ROLE` is also its own admin: it has permission to
 * grant and revoke this role. Extra precautions should be taken to secure
 * accounts that have been granted it. We recommend using {AccessControlDefaultAdminRules}
 * to enforce additional security measures for this role.
 */
abstract contract AccessControlUpgradeable is Initializable, ContextUpgradeable, IAccessControl, ERC165Upgradeable {
    struct RoleData {
        mapping(address account => bool) hasRole;
        bytes32 adminRole;
    }

    bytes32 public constant DEFAULT_ADMIN_ROLE = 0x00;


    /// @custom:storage-location erc7201:openzeppelin.storage.AccessControl
    struct AccessControlStorage {
        mapping(bytes32 role => RoleData) _roles;
    }

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.AccessControl")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant AccessControlStorageLocation = 0x02dd7bc7dec4dceedda775e58dd541e08a116c6c53815c0bd028192f7b626800;

    function _getAccessControlStorage() private pure returns (AccessControlStorage storage $) {
        assembly {
            $.slot := AccessControlStorageLocation
        }
    }

    /**
     * @dev Modifier that checks that an account has a specific role. Reverts
     * with an {AccessControlUnauthorizedAccount} error including the required role.
     */
    modifier onlyRole(bytes32 role) {
        _checkRole(role);
        _;
    }

    function __AccessControl_init() internal onlyInitializing {
    }

    function __AccessControl_init_unchained() internal onlyInitializing {
    }
    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId) public view virtual override returns (bool) {
        return interfaceId == type(IAccessControl).interfaceId || super.supportsInterface(interfaceId);
    }

    /**
     * @dev Returns `true` if `account` has been granted `role`.
     */
    function hasRole(bytes32 role, address account) public view virtual returns (bool) {
        AccessControlStorage storage $ = _getAccessControlStorage();
        return $._roles[role].hasRole[account];
    }

    /**
     * @dev Reverts with an {AccessControlUnauthorizedAccount} error if `_msgSender()`
     * is missing `role`. Overriding this function changes the behavior of the {onlyRole} modifier.
     */
    function _checkRole(bytes32 role) internal view virtual {
        _checkRole(role, _msgSender());
    }

    /**
     * @dev Reverts with an {AccessControlUnauthorizedAccount} error if `account`
     * is missing `role`.
     */
    function _checkRole(bytes32 role, address account) internal view virtual {
        if (!hasRole(role, account)) {
            revert AccessControlUnauthorizedAccount(account, role);
        }
    }

    /**
     * @dev Returns the admin role that controls `role`. See {grantRole} and
     * {revokeRole}.
     *
     * To change a role's admin, use {_setRoleAdmin}.
     */
    function getRoleAdmin(bytes32 role) public view virtual returns (bytes32) {
        AccessControlStorage storage $ = _getAccessControlStorage();
        return $._roles[role].adminRole;
    }

    /**
     * @dev Grants `role` to `account`.
     *
     * If `account` had not been already granted `role`, emits a {RoleGranted}
     * event.
     *
     * Requirements:
     *
     * - the caller must have ``role``'s admin role.
     *
     * May emit a {RoleGranted} event.
     */
    function grantRole(bytes32 role, address account) public virtual onlyRole(getRoleAdmin(role)) {
        _grantRole(role, account);
    }

    /**
     * @dev Revokes `role` from `account`.
     *
     * If `account` had been granted `role`, emits a {RoleRevoked} event.
     *
     * Requirements:
     *
     * - the caller must have ``role``'s admin role.
     *
     * May emit a {RoleRevoked} event.
     */
    function revokeRole(bytes32 role, address account) public virtual onlyRole(getRoleAdmin(role)) {
        _revokeRole(role, account);
    }

    /**
     * @dev Revokes `role` from the calling account.
     *
     * Roles are often managed via {grantRole} and {revokeRole}: this function's
     * purpose is to provide a mechanism for accounts to lose their privileges
     * if they are compromised (such as when a trusted device is misplaced).
     *
     * If the calling account had been revoked `role`, emits a {RoleRevoked}
     * event.
     *
     * Requirements:
     *
     * - the caller must be `callerConfirmation`.
     *
     * May emit a {RoleRevoked} event.
     */
    function renounceRole(bytes32 role, address callerConfirmation) public virtual {
        if (callerConfirmation != _msgSender()) {
            revert AccessControlBadConfirmation();
        }

        _revokeRole(role, callerConfirmation);
    }

    /**
     * @dev Sets `adminRole` as ``role``'s admin role.
     *
     * Emits a {RoleAdminChanged} event.
     */
    function _setRoleAdmin(bytes32 role, bytes32 adminRole) internal virtual {
        AccessControlStorage storage $ = _getAccessControlStorage();
        bytes32 previousAdminRole = getRoleAdmin(role);
        $._roles[role].adminRole = adminRole;
        emit RoleAdminChanged(role, previousAdminRole, adminRole);
    }

    /**
     * @dev Attempts to grant `role` to `account` and returns a boolean indicating if `role` was granted.
     *
     * Internal function without access restriction.
     *
     * May emit a {RoleGranted} event.
     */
    function _grantRole(bytes32 role, address account) internal virtual returns (bool) {
        AccessControlStorage storage $ = _getAccessControlStorage();
        if (!hasRole(role, account)) {
            $._roles[role].hasRole[account] = true;
            emit RoleGranted(role, account, _msgSender());
            return true;
        } else {
            return false;
        }
    }

    /**
     * @dev Attempts to revoke `role` to `account` and returns a boolean indicating if `role` was revoked.
     *
     * Internal function without access restriction.
     *
     * May emit a {RoleRevoked} event.
     */
    function _revokeRole(bytes32 role, address account) internal virtual returns (bool) {
        AccessControlStorage storage $ = _getAccessControlStorage();
        if (hasRole(role, account)) {
            $._roles[role].hasRole[account] = false;
            emit RoleRevoked(role, account, _msgSender());
            return true;
        } else {
            return false;
        }
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (proxy/utils/Initializable.sol)

pragma solidity ^0.8.20;

/**
 * @dev This is a base contract to aid in writing upgradeable contracts, or any kind of contract that will be deployed
 * behind a proxy. Since proxied contracts do not make use of a constructor, it's common to move constructor logic to an
 * external initializer function, usually called `initialize`. It then becomes necessary to protect this initializer
 * function so it can only be called once. The {initializer} modifier provided by this contract will have this effect.
 *
 * The initialization functions use a version number. Once a version number is used, it is consumed and cannot be
 * reused. This mechanism prevents re-execution of each "step" but allows the creation of new initialization steps in
 * case an upgrade adds a module that needs to be initialized.
 *
 * For example:
 *
 * [.hljs-theme-light.nopadding]
 * ```solidity
 * contract MyToken is ERC20Upgradeable {
 *     function initialize() initializer public {
 *         __ERC20_init("MyToken", "MTK");
 *     }
 * }
 *
 * contract MyTokenV2 is MyToken, ERC20PermitUpgradeable {
 *     function initializeV2() reinitializer(2) public {
 *         __ERC20Permit_init("MyToken");
 *     }
 * }
 * ```
 *
 * TIP: To avoid leaving the proxy in an uninitialized state, the initializer function should be called as early as
 * possible by providing the encoded function call as the `_data` argument to {ERC1967Proxy-constructor}.
 *
 * CAUTION: When used with inheritance, manual care must be taken to not invoke a parent initializer twice, or to ensure
 * that all initializers are idempotent. This is not verified automatically as constructors are by Solidity.
 *
 * [CAUTION]
 * ====
 * Avoid leaving a contract uninitialized.
 *
 * An uninitialized contract can be taken over by an attacker. This applies to both a proxy and its implementation
 * contract, which may impact the proxy. To prevent the implementation contract from being used, you should invoke
 * the {_disableInitializers} function in the constructor to automatically lock it when it is deployed:
 *
 * [.hljs-theme-light.nopadding]
 * ```
 * /// @custom:oz-upgrades-unsafe-allow constructor
 * constructor() {
 *     _disableInitializers();
 * }
 * ```
 * ====
 */
abstract contract Initializable {
    /**
     * @dev Storage of the initializable contract.
     *
     * It's implemented on a custom ERC-7201 namespace to reduce the risk of storage collisions
     * when using with upgradeable contracts.
     *
     * @custom:storage-location erc7201:openzeppelin.storage.Initializable
     */
    struct InitializableStorage {
        /**
         * @dev Indicates that the contract has been initialized.
         */
        uint64 _initialized;
        /**
         * @dev Indicates that the contract is in the process of being initialized.
         */
        bool _initializing;
    }

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.Initializable")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant INITIALIZABLE_STORAGE = 0xf0c57e16840df040f15088dc2f81fe391c3923bec73e23a9662efc9c229c6a00;

    /**
     * @dev The contract is already initialized.
     */
    error InvalidInitialization();

    /**
     * @dev The contract is not initializing.
     */
    error NotInitializing();

    /**
     * @dev Triggered when the contract has been initialized or reinitialized.
     */
    event Initialized(uint64 version);

    /**
     * @dev A modifier that defines a protected initializer function that can be invoked at most once. In its scope,
     * `onlyInitializing` functions can be used to initialize parent contracts.
     *
     * Similar to `reinitializer(1)`, except that in the context of a constructor an `initializer` may be invoked any
     * number of times. This behavior in the constructor can be useful during testing and is not expected to be used in
     * production.
     *
     * Emits an {Initialized} event.
     */
    modifier initializer() {
        // solhint-disable-next-line var-name-mixedcase
        InitializableStorage storage $ = _getInitializableStorage();

        // Cache values to avoid duplicated sloads
        bool isTopLevelCall = !$._initializing;
        uint64 initialized = $._initialized;

        // Allowed calls:
        // - initialSetup: the contract is not in the initializing state and no previous version was
        //                 initialized
        // - construction: the contract is initialized at version 1 (no reininitialization) and the
        //                 current contract is just being deployed
        bool initialSetup = initialized == 0 && isTopLevelCall;
        bool construction = initialized == 1 && address(this).code.length == 0;

        if (!initialSetup && !construction) {
            revert InvalidInitialization();
        }
        $._initialized = 1;
        if (isTopLevelCall) {
            $._initializing = true;
        }
        _;
        if (isTopLevelCall) {
            $._initializing = false;
            emit Initialized(1);
        }
    }

    /**
     * @dev A modifier that defines a protected reinitializer function that can be invoked at most once, and only if the
     * contract hasn't been initialized to a greater version before. In its scope, `onlyInitializing` functions can be
     * used to initialize parent contracts.
     *
     * A reinitializer may be used after the original initialization step. This is essential to configure modules that
     * are added through upgrades and that require initialization.
     *
     * When `version` is 1, this modifier is similar to `initializer`, except that functions marked with `reinitializer`
     * cannot be nested. If one is invoked in the context of another, execution will revert.
     *
     * Note that versions can jump in increments greater than 1; this implies that if multiple reinitializers coexist in
     * a contract, executing them in the right order is up to the developer or operator.
     *
     * WARNING: Setting the version to 2**64 - 1 will prevent any future reinitialization.
     *
     * Emits an {Initialized} event.
     */
    modifier reinitializer(uint64 version) {
        // solhint-disable-next-line var-name-mixedcase
        InitializableStorage storage $ = _getInitializableStorage();

        if ($._initializing || $._initialized >= version) {
            revert InvalidInitialization();
        }
        $._initialized = version;
        $._initializing = true;
        _;
        $._initializing = false;
        emit Initialized(version);
    }

    /**
     * @dev Modifier to protect an initialization function so that it can only be invoked by functions with the
     * {initializer} and {reinitializer} modifiers, directly or indirectly.
     */
    modifier onlyInitializing() {
        _checkInitializing();
        _;
    }

    /**
     * @dev Reverts if the contract is not in an initializing state. See {onlyInitializing}.
     */
    function _checkInitializing() internal view virtual {
        if (!_isInitializing()) {
            revert NotInitializing();
        }
    }

    /**
     * @dev Locks the contract, preventing any future reinitialization. This cannot be part of an initializer call.
     * Calling this in the constructor of a contract will prevent that contract from being initialized or reinitialized
     * to any version. It is recommended to use this to lock implementation contracts that are designed to be called
     * through proxies.
     *
     * Emits an {Initialized} event the first time it is successfully executed.
     */
    function _disableInitializers() internal virtual {
        // solhint-disable-next-line var-name-mixedcase
        InitializableStorage storage $ = _getInitializableStorage();

        if ($._initializing) {
            revert InvalidInitialization();
        }
        if ($._initialized != type(uint64).max) {
            $._initialized = type(uint64).max;
            emit Initialized(type(uint64).max);
        }
    }

    /**
     * @dev Returns the highest version that has been initialized. See {reinitializer}.
     */
    function _getInitializedVersion() internal view returns (uint64) {
        return _getInitializableStorage()._initialized;
    }

    /**
     * @dev Returns `true` if the contract is currently initializing. See {onlyInitializing}.
     */
    function _isInitializing() internal view returns (bool) {
        return _getInitializableStorage()._initializing;
    }

    /**
     * @dev Returns a pointer to the storage namespace.
     */
    // solhint-disable-next-line var-name-mixedcase
    function _getInitializableStorage() private pure returns (InitializableStorage storage $) {
        assembly {
            $.slot := INITIALIZABLE_STORAGE
        }
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts/token/ERC20/IERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (token/ERC20/IERC20.sol)

pragma solidity ^0.8.20;

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


// ===== FILE: node_modules/_openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.2.0) (token/ERC20/utils/SafeERC20.sol)

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


// ===== FILE: node_modules/_openzeppelin/contracts/interfaces/IERC4626.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (interfaces/IERC4626.sol)

pragma solidity ^0.8.20;

import {IERC20} from "../token/ERC20/IERC20.sol";
import {IERC20Metadata} from "../token/ERC20/extensions/IERC20Metadata.sol";

/**
 * @dev Interface of the ERC-4626 "Tokenized Vault Standard", as defined in
 * https://eips.ethereum.org/EIPS/eip-4626[ERC-4626].
 */
interface IERC4626 is IERC20, IERC20Metadata {
    event Deposit(address indexed sender, address indexed owner, uint256 assets, uint256 shares);

    event Withdraw(
        address indexed sender,
        address indexed receiver,
        address indexed owner,
        uint256 assets,
        uint256 shares
    );

    /**
     * @dev Returns the address of the underlying token used for the Vault for accounting, depositing, and withdrawing.
     *
     * - MUST be an ERC-20 token contract.
     * - MUST NOT revert.
     */
    function asset() external view returns (address assetTokenAddress);

    /**
     * @dev Returns the total amount of the underlying asset that is “managed” by Vault.
     *
     * - SHOULD include any compounding that occurs from yield.
     * - MUST be inclusive of any fees that are charged against assets in the Vault.
     * - MUST NOT revert.
     */
    function totalAssets() external view returns (uint256 totalManagedAssets);

    /**
     * @dev Returns the amount of shares that the Vault would exchange for the amount of assets provided, in an ideal
     * scenario where all the conditions are met.
     *
     * - MUST NOT be inclusive of any fees that are charged against assets in the Vault.
     * - MUST NOT show any variations depending on the caller.
     * - MUST NOT reflect slippage or other on-chain conditions, when performing the actual exchange.
     * - MUST NOT revert.
     *
     * NOTE: This calculation MAY NOT reflect the “per-user” price-per-share, and instead should reflect the
     * “average-user’s” price-per-share, meaning what the average user should expect to see when exchanging to and
     * from.
     */
    function convertToShares(uint256 assets) external view returns (uint256 shares);

    /**
     * @dev Returns the amount of assets that the Vault would exchange for the amount of shares provided, in an ideal
     * scenario where all the conditions are met.
     *
     * - MUST NOT be inclusive of any fees that are charged against assets in the Vault.
     * - MUST NOT show any variations depending on the caller.
     * - MUST NOT reflect slippage or other on-chain conditions, when performing the actual exchange.
     * - MUST NOT revert.
     *
     * NOTE: This calculation MAY NOT reflect the “per-user” price-per-share, and instead should reflect the
     * “average-user’s” price-per-share, meaning what the average user should expect to see when exchanging to and
     * from.
     */
    function convertToAssets(uint256 shares) external view returns (uint256 assets);

    /**
     * @dev Returns the maximum amount of the underlying asset that can be deposited into the Vault for the receiver,
     * through a deposit call.
     *
     * - MUST return a limited value if receiver is subject to some deposit limit.
     * - MUST return 2 ** 256 - 1 if there is no limit on the maximum amount of assets that may be deposited.
     * - MUST NOT revert.
     */
    function maxDeposit(address receiver) external view returns (uint256 maxAssets);

    /**
     * @dev Allows an on-chain or off-chain user to simulate the effects of their deposit at the current block, given
     * current on-chain conditions.
     *
     * - MUST return as close to and no more than the exact amount of Vault shares that would be minted in a deposit
     *   call in the same transaction. I.e. deposit should return the same or more shares as previewDeposit if called
     *   in the same transaction.
     * - MUST NOT account for deposit limits like those returned from maxDeposit and should always act as though the
     *   deposit would be accepted, regardless if the user has enough tokens approved, etc.
     * - MUST be inclusive of deposit fees. Integrators should be aware of the existence of deposit fees.
     * - MUST NOT revert.
     *
     * NOTE: any unfavorable discrepancy between convertToShares and previewDeposit SHOULD be considered slippage in
     * share price or some other type of condition, meaning the depositor will lose assets by depositing.
     */
    function previewDeposit(uint256 assets) external view returns (uint256 shares);

    /**
     * @dev Mints shares Vault shares to receiver by depositing exactly amount of underlying tokens.
     *
     * - MUST emit the Deposit event.
     * - MAY support an additional flow in which the underlying tokens are owned by the Vault contract before the
     *   deposit execution, and are accounted for during deposit.
     * - MUST revert if all of assets cannot be deposited (due to deposit limit being reached, slippage, the user not
     *   approving enough underlying tokens to the Vault contract, etc).
     *
     * NOTE: most implementations will require pre-approval of the Vault with the Vault’s underlying asset token.
     */
    function deposit(uint256 assets, address receiver) external returns (uint256 shares);

    /**
     * @dev Returns the maximum amount of the Vault shares that can be minted for the receiver, through a mint call.
     * - MUST return a limited value if receiver is subject to some mint limit.
     * - MUST return 2 ** 256 - 1 if there is no limit on the maximum amount of shares that may be minted.
     * - MUST NOT revert.
     */
    function maxMint(address receiver) external view returns (uint256 maxShares);

    /**
     * @dev Allows an on-chain or off-chain user to simulate the effects of their mint at the current block, given
     * current on-chain conditions.
     *
     * - MUST return as close to and no fewer than the exact amount of assets that would be deposited in a mint call
     *   in the same transaction. I.e. mint should return the same or fewer assets as previewMint if called in the
     *   same transaction.
     * - MUST NOT account for mint limits like those returned from maxMint and should always act as though the mint
     *   would be accepted, regardless if the user has enough tokens approved, etc.
     * - MUST be inclusive of deposit fees. Integrators should be aware of the existence of deposit fees.
     * - MUST NOT revert.
     *
     * NOTE: any unfavorable discrepancy between convertToAssets and previewMint SHOULD be considered slippage in
     * share price or some other type of condition, meaning the depositor will lose assets by minting.
     */
    function previewMint(uint256 shares) external view returns (uint256 assets);

    /**
     * @dev Mints exactly shares Vault shares to receiver by depositing amount of underlying tokens.
     *
     * - MUST emit the Deposit event.
     * - MAY support an additional flow in which the underlying tokens are owned by the Vault contract before the mint
     *   execution, and are accounted for during mint.
     * - MUST revert if all of shares cannot be minted (due to deposit limit being reached, slippage, the user not
     *   approving enough underlying tokens to the Vault contract, etc).
     *
     * NOTE: most implementations will require pre-approval of the Vault with the Vault’s underlying asset token.
     */
    function mint(uint256 shares, address receiver) external returns (uint256 assets);

    /**
     * @dev Returns the maximum amount of the underlying asset that can be withdrawn from the owner balance in the
     * Vault, through a withdraw call.
     *
     * - MUST return a limited value if owner is subject to some withdrawal limit or timelock.
     * - MUST NOT revert.
     */
    function maxWithdraw(address owner) external view returns (uint256 maxAssets);

    /**
     * @dev Allows an on-chain or off-chain user to simulate the effects of their withdrawal at the current block,
     * given current on-chain conditions.
     *
     * - MUST return as close to and no fewer than the exact amount of Vault shares that would be burned in a withdraw
     *   call in the same transaction. I.e. withdraw should return the same or fewer shares as previewWithdraw if
     *   called
     *   in the same transaction.
     * - MUST NOT account for withdrawal limits like those returned from maxWithdraw and should always act as though
     *   the withdrawal would be accepted, regardless if the user has enough shares, etc.
     * - MUST be inclusive of withdrawal fees. Integrators should be aware of the existence of withdrawal fees.
     * - MUST NOT revert.
     *
     * NOTE: any unfavorable discrepancy between convertToShares and previewWithdraw SHOULD be considered slippage in
     * share price or some other type of condition, meaning the depositor will lose assets by depositing.
     */
    function previewWithdraw(uint256 assets) external view returns (uint256 shares);

    /**
     * @dev Burns shares from owner and sends exactly assets of underlying tokens to receiver.
     *
     * - MUST emit the Withdraw event.
     * - MAY support an additional flow in which the underlying tokens are owned by the Vault contract before the
     *   withdraw execution, and are accounted for during withdraw.
     * - MUST revert if all of assets cannot be withdrawn (due to withdrawal limit being reached, slippage, the owner
     *   not having enough shares, etc).
     *
     * Note that some implementations will require pre-requesting to the Vault before a withdrawal may be performed.
     * Those methods should be performed separately.
     */
    function withdraw(uint256 assets, address receiver, address owner) external returns (uint256 shares);

    /**
     * @dev Returns the maximum amount of Vault shares that can be redeemed from the owner balance in the Vault,
     * through a redeem call.
     *
     * - MUST return a limited value if owner is subject to some withdrawal limit or timelock.
     * - MUST return balanceOf(owner) if owner is not subject to any withdrawal limit or timelock.
     * - MUST NOT revert.
     */
    function maxRedeem(address owner) external view returns (uint256 maxShares);

    /**
     * @dev Allows an on-chain or off-chain user to simulate the effects of their redeemption at the current block,
     * given current on-chain conditions.
     *
     * - MUST return as close to and no more than the exact amount of assets that would be withdrawn in a redeem call
     *   in the same transaction. I.e. redeem should return the same or more assets as previewRedeem if called in the
     *   same transaction.
     * - MUST NOT account for redemption limits like those returned from maxRedeem and should always act as though the
     *   redemption would be accepted, regardless if the user has enough shares, etc.
     * - MUST be inclusive of withdrawal fees. Integrators should be aware of the existence of withdrawal fees.
     * - MUST NOT revert.
     *
     * NOTE: any unfavorable discrepancy between convertToAssets and previewRedeem SHOULD be considered slippage in
     * share price or some other type of condition, meaning the depositor will lose assets by redeeming.
     */
    function previewRedeem(uint256 shares) external view returns (uint256 assets);

    /**
     * @dev Burns exactly shares from owner and sends assets of underlying tokens to receiver.
     *
     * - MUST emit the Withdraw event.
     * - MAY support an additional flow in which the underlying tokens are owned by the Vault contract before the
     *   redeem execution, and are accounted for during redeem.
     * - MUST revert if all of shares cannot be redeemed (due to withdrawal limit being reached, slippage, the owner
     *   not having enough shares, etc).
     *
     * NOTE: some implementations will require pre-requesting to the Vault before a withdrawal may be performed.
     * Those methods should be performed separately.
     */
    function redeem(uint256 shares, address receiver, address owner) external returns (uint256 assets);
}


// ===== FILE: node_modules/_openzeppelin/contracts/utils/math/Math.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/math/Math.sol)

pragma solidity ^0.8.20;

import {Panic} from "../Panic.sol";
import {SafeCast} from "./SafeCast.sol";

/**
 * @dev Standard math utilities missing in the Solidity language.
 */
library Math {
    enum Rounding {
        Floor, // Toward negative infinity
        Ceil, // Toward positive infinity
        Trunc, // Toward zero
        Expand // Away from zero
    }

    /**
     * @dev Returns the addition of two unsigned integers, with an success flag (no overflow).
     */
    function tryAdd(uint256 a, uint256 b) internal pure returns (bool success, uint256 result) {
        unchecked {
            uint256 c = a + b;
            if (c < a) return (false, 0);
            return (true, c);
        }
    }

    /**
     * @dev Returns the subtraction of two unsigned integers, with an success flag (no overflow).
     */
    function trySub(uint256 a, uint256 b) internal pure returns (bool success, uint256 result) {
        unchecked {
            if (b > a) return (false, 0);
            return (true, a - b);
        }
    }

    /**
     * @dev Returns the multiplication of two unsigned integers, with an success flag (no overflow).
     */
    function tryMul(uint256 a, uint256 b) internal pure returns (bool success, uint256 result) {
        unchecked {
            // Gas optimization: this is cheaper than requiring 'a' not being zero, but the
            // benefit is lost if 'b' is also tested.
            // See: https://github.com/OpenZeppelin/openzeppelin-contracts/pull/522
            if (a == 0) return (true, 0);
            uint256 c = a * b;
            if (c / a != b) return (false, 0);
            return (true, c);
        }
    }

    /**
     * @dev Returns the division of two unsigned integers, with a success flag (no division by zero).
     */
    function tryDiv(uint256 a, uint256 b) internal pure returns (bool success, uint256 result) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a / b);
        }
    }

    /**
     * @dev Returns the remainder of dividing two unsigned integers, with a success flag (no division by zero).
     */
    function tryMod(uint256 a, uint256 b) internal pure returns (bool success, uint256 result) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a % b);
        }
    }

    /**
     * @dev Branchless ternary evaluation for `a ? b : c`. Gas costs are constant.
     *
     * IMPORTANT: This function may reduce bytecode size and consume less gas when used standalone.
     * However, the compiler may optimize Solidity ternary operations (i.e. `a ? b : c`) to only compute
     * one branch when needed, making this function more expensive.
     */
    function ternary(bool condition, uint256 a, uint256 b) internal pure returns (uint256) {
        unchecked {
            // branchless ternary works because:
            // b ^ (a ^ b) == a
            // b ^ 0 == b
            return b ^ ((a ^ b) * SafeCast.toUint(condition));
        }
    }

    /**
     * @dev Returns the largest of two numbers.
     */
    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return ternary(a > b, a, b);
    }

    /**
     * @dev Returns the smallest of two numbers.
     */
    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return ternary(a < b, a, b);
    }

    /**
     * @dev Returns the average of two numbers. The result is rounded towards
     * zero.
     */
    function average(uint256 a, uint256 b) internal pure returns (uint256) {
        // (a + b) / 2 can overflow.
        return (a & b) + (a ^ b) / 2;
    }

    /**
     * @dev Returns the ceiling of the division of two numbers.
     *
     * This differs from standard division with `/` in that it rounds towards infinity instead
     * of rounding towards zero.
     */
    function ceilDiv(uint256 a, uint256 b) internal pure returns (uint256) {
        if (b == 0) {
            // Guarantee the same behavior as in a regular Solidity division.
            Panic.panic(Panic.DIVISION_BY_ZERO);
        }

        // The following calculation ensures accurate ceiling division without overflow.
        // Since a is non-zero, (a - 1) / b will not overflow.
        // The largest possible result occurs when (a - 1) / b is type(uint256).max,
        // but the largest value we can obtain is type(uint256).max - 1, which happens
        // when a = type(uint256).max and b = 1.
        unchecked {
            return SafeCast.toUint(a > 0) * ((a - 1) / b + 1);
        }
    }

    /**
     * @dev Calculates floor(x * y / denominator) with full precision. Throws if result overflows a uint256 or
     * denominator == 0.
     *
     * Original credit to Remco Bloemen under MIT license (https://xn--2-umb.com/21/muldiv) with further edits by
     * Uniswap Labs also under MIT license.
     */
    function mulDiv(uint256 x, uint256 y, uint256 denominator) internal pure returns (uint256 result) {
        unchecked {
            // 512-bit multiply [prod1 prod0] = x * y. Compute the product mod 2²⁵⁶ and mod 2²⁵⁶ - 1, then use
            // the Chinese Remainder Theorem to reconstruct the 512 bit result. The result is stored in two 256
            // variables such that product = prod1 * 2²⁵⁶ + prod0.
            uint256 prod0 = x * y; // Least significant 256 bits of the product
            uint256 prod1; // Most significant 256 bits of the product
            assembly {
                let mm := mulmod(x, y, not(0))
                prod1 := sub(sub(mm, prod0), lt(mm, prod0))
            }

            // Handle non-overflow cases, 256 by 256 division.
            if (prod1 == 0) {
                // Solidity will revert if denominator == 0, unlike the div opcode on its own.
                // The surrounding unchecked block does not change this fact.
                // See https://docs.soliditylang.org/en/latest/control-structures.html#checked-or-unchecked-arithmetic.
                return prod0 / denominator;
            }

            // Make sure the result is less than 2²⁵⁶. Also prevents denominator == 0.
            if (denominator <= prod1) {
                Panic.panic(ternary(denominator == 0, Panic.DIVISION_BY_ZERO, Panic.UNDER_OVERFLOW));
            }

            ///////////////////////////////////////////////
            // 512 by 256 division.
            ///////////////////////////////////////////////

            // Make division exact by subtracting the remainder from [prod1 prod0].
            uint256 remainder;
            assembly {
                // Compute remainder using mulmod.
                remainder := mulmod(x, y, denominator)

                // Subtract 256 bit number from 512 bit number.
                prod1 := sub(prod1, gt(remainder, prod0))
                prod0 := sub(prod0, remainder)
            }

            // Factor powers of two out of denominator and compute largest power of two divisor of denominator.
            // Always >= 1. See https://cs.stackexchange.com/q/138556/92363.

            uint256 twos = denominator & (0 - denominator);
            assembly {
                // Divide denominator by twos.
                denominator := div(denominator, twos)

                // Divide [prod1 prod0] by twos.
                prod0 := div(prod0, twos)

                // Flip twos such that it is 2²⁵⁶ / twos. If twos is zero, then it becomes one.
                twos := add(div(sub(0, twos), twos), 1)
            }

            // Shift in bits from prod1 into prod0.
            prod0 |= prod1 * twos;

            // Invert denominator mod 2²⁵⁶. Now that denominator is an odd number, it has an inverse modulo 2²⁵⁶ such
            // that denominator * inv ≡ 1 mod 2²⁵⁶. Compute the inverse by starting with a seed that is correct for
            // four bits. That is, denominator * inv ≡ 1 mod 2⁴.
            uint256 inverse = (3 * denominator) ^ 2;

            // Use the Newton-Raphson iteration to improve the precision. Thanks to Hensel's lifting lemma, this also
            // works in modular arithmetic, doubling the correct bits in each step.
            inverse *= 2 - denominator * inverse; // inverse mod 2⁸
            inverse *= 2 - denominator * inverse; // inverse mod 2¹⁶
            inverse *= 2 - denominator * inverse; // inverse mod 2³²
            inverse *= 2 - denominator * inverse; // inverse mod 2⁶⁴
            inverse *= 2 - denominator * inverse; // inverse mod 2¹²⁸
            inverse *= 2 - denominator * inverse; // inverse mod 2²⁵⁶

            // Because the division is now exact we can divide by multiplying with the modular inverse of denominator.
            // This will give us the correct result modulo 2²⁵⁶. Since the preconditions guarantee that the outcome is
            // less than 2²⁵⁶, this is the final result. We don't need to compute the high bits of the result and prod1
            // is no longer required.
            result = prod0 * inverse;
            return result;
        }
    }

    /**
     * @dev Calculates x * y / denominator with full precision, following the selected rounding direction.
     */
    function mulDiv(uint256 x, uint256 y, uint256 denominator, Rounding rounding) internal pure returns (uint256) {
        return mulDiv(x, y, denominator) + SafeCast.toUint(unsignedRoundsUp(rounding) && mulmod(x, y, denominator) > 0);
    }

    /**
     * @dev Calculate the modular multiplicative inverse of a number in Z/nZ.
     *
     * If n is a prime, then Z/nZ is a field. In that case all elements are inversible, except 0.
     * If n is not a prime, then Z/nZ is not a field, and some elements might not be inversible.
     *
     * If the input value is not inversible, 0 is returned.
     *
     * NOTE: If you know for sure that n is (big) a prime, it may be cheaper to use Fermat's little theorem and get the
     * inverse using `Math.modExp(a, n - 2, n)`. See {invModPrime}.
     */
    function invMod(uint256 a, uint256 n) internal pure returns (uint256) {
        unchecked {
            if (n == 0) return 0;

            // The inverse modulo is calculated using the Extended Euclidean Algorithm (iterative version)
            // Used to compute integers x and y such that: ax + ny = gcd(a, n).
            // When the gcd is 1, then the inverse of a modulo n exists and it's x.
            // ax + ny = 1
            // ax = 1 + (-y)n
            // ax ≡ 1 (mod n) # x is the inverse of a modulo n

            // If the remainder is 0 the gcd is n right away.
            uint256 remainder = a % n;
            uint256 gcd = n;

            // Therefore the initial coefficients are:
            // ax + ny = gcd(a, n) = n
            // 0a + 1n = n
            int256 x = 0;
            int256 y = 1;

            while (remainder != 0) {
                uint256 quotient = gcd / remainder;

                (gcd, remainder) = (
                    // The old remainder is the next gcd to try.
                    remainder,
                    // Compute the next remainder.
                    // Can't overflow given that (a % gcd) * (gcd // (a % gcd)) <= gcd
                    // where gcd is at most n (capped to type(uint256).max)
                    gcd - remainder * quotient
                );

                (x, y) = (
                    // Increment the coefficient of a.
                    y,
                    // Decrement the coefficient of n.
                    // Can overflow, but the result is casted to uint256 so that the
                    // next value of y is "wrapped around" to a value between 0 and n - 1.
                    x - y * int256(quotient)
                );
            }

            if (gcd != 1) return 0; // No inverse exists.
            return ternary(x < 0, n - uint256(-x), uint256(x)); // Wrap the result if it's negative.
        }
    }

    /**
     * @dev Variant of {invMod}. More efficient, but only works if `p` is known to be a prime greater than `2`.
     *
     * From https://en.wikipedia.org/wiki/Fermat%27s_little_theorem[Fermat's little theorem], we know that if p is
     * prime, then `a**(p-1) ≡ 1 mod p`. As a consequence, we have `a * a**(p-2) ≡ 1 mod p`, which means that
     * `a**(p-2)` is the modular multiplicative inverse of a in Fp.
     *
     * NOTE: this function does NOT check that `p` is a prime greater than `2`.
     */
    function invModPrime(uint256 a, uint256 p) internal view returns (uint256) {
        unchecked {
            return Math.modExp(a, p - 2, p);
        }
    }

    /**
     * @dev Returns the modular exponentiation of the specified base, exponent and modulus (b ** e % m)
     *
     * Requirements:
     * - modulus can't be zero
     * - underlying staticcall to precompile must succeed
     *
     * IMPORTANT: The result is only valid if the underlying call succeeds. When using this function, make
     * sure the chain you're using it on supports the precompiled contract for modular exponentiation
     * at address 0x05 as specified in https://eips.ethereum.org/EIPS/eip-198[EIP-198]. Otherwise,
     * the underlying function will succeed given the lack of a revert, but the result may be incorrectly
     * interpreted as 0.
     */
    function modExp(uint256 b, uint256 e, uint256 m) internal view returns (uint256) {
        (bool success, uint256 result) = tryModExp(b, e, m);
        if (!success) {
            Panic.panic(Panic.DIVISION_BY_ZERO);
        }
        return result;
    }

    /**
     * @dev Returns the modular exponentiation of the specified base, exponent and modulus (b ** e % m).
     * It includes a success flag indicating if the operation succeeded. Operation will be marked as failed if trying
     * to operate modulo 0 or if the underlying precompile reverted.
     *
     * IMPORTANT: The result is only valid if the success flag is true. When using this function, make sure the chain
     * you're using it on supports the precompiled contract for modular exponentiation at address 0x05 as specified in
     * https://eips.ethereum.org/EIPS/eip-198[EIP-198]. Otherwise, the underlying function will succeed given the lack
     * of a revert, but the result may be incorrectly interpreted as 0.
     */
    function tryModExp(uint256 b, uint256 e, uint256 m) internal view returns (bool success, uint256 result) {
        if (m == 0) return (false, 0);
        assembly ("memory-safe") {
            let ptr := mload(0x40)
            // | Offset    | Content    | Content (Hex)                                                      |
            // |-----------|------------|--------------------------------------------------------------------|
            // | 0x00:0x1f | size of b  | 0x0000000000000000000000000000000000000000000000000000000000000020 |
            // | 0x20:0x3f | size of e  | 0x0000000000000000000000000000000000000000000000000000000000000020 |
            // | 0x40:0x5f | size of m  | 0x0000000000000000000000000000000000000000000000000000000000000020 |
            // | 0x60:0x7f | value of b | 0x<.............................................................b> |
            // | 0x80:0x9f | value of e | 0x<.............................................................e> |
            // | 0xa0:0xbf | value of m | 0x<.............................................................m> |
            mstore(ptr, 0x20)
            mstore(add(ptr, 0x20), 0x20)
            mstore(add(ptr, 0x40), 0x20)
            mstore(add(ptr, 0x60), b)
            mstore(add(ptr, 0x80), e)
            mstore(add(ptr, 0xa0), m)

            // Given the result < m, it's guaranteed to fit in 32 bytes,
            // so we can use the memory scratch space located at offset 0.
            success := staticcall(gas(), 0x05, ptr, 0xc0, 0x00, 0x20)
            result := mload(0x00)
        }
    }

    /**
     * @dev Variant of {modExp} that supports inputs of arbitrary length.
     */
    function modExp(bytes memory b, bytes memory e, bytes memory m) internal view returns (bytes memory) {
        (bool success, bytes memory result) = tryModExp(b, e, m);
        if (!success) {
            Panic.panic(Panic.DIVISION_BY_ZERO);
        }
        return result;
    }

    /**
     * @dev Variant of {tryModExp} that supports inputs of arbitrary length.
     */
    function tryModExp(
        bytes memory b,
        bytes memory e,
        bytes memory m
    ) internal view returns (bool success, bytes memory result) {
        if (_zeroBytes(m)) return (false, new bytes(0));

        uint256 mLen = m.length;

        // Encode call args in result and move the free memory pointer
        result = abi.encodePacked(b.length, e.length, mLen, b, e, m);

        assembly ("memory-safe") {
            let dataPtr := add(result, 0x20)
            // Write result on top of args to avoid allocating extra memory.
            success := staticcall(gas(), 0x05, dataPtr, mload(result), dataPtr, mLen)
            // Overwrite the length.
            // result.length > returndatasize() is guaranteed because returndatasize() == m.length
            mstore(result, mLen)
            // Set the memory pointer after the returned data.
            mstore(0x40, add(dataPtr, mLen))
        }
    }

    /**
     * @dev Returns whether the provided byte array is zero.
     */
    function _zeroBytes(bytes memory byteArray) private pure returns (bool) {
        for (uint256 i = 0; i < byteArray.length; ++i) {
            if (byteArray[i] != 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * @dev Returns the square root of a number. If the number is not a perfect square, the value is rounded
     * towards zero.
     *
     * This method is based on Newton's method for computing square roots; the algorithm is restricted to only
     * using integer operations.
     */
    function sqrt(uint256 a) internal pure returns (uint256) {
        unchecked {
            // Take care of easy edge cases when a == 0 or a == 1
            if (a <= 1) {
                return a;
            }

            // In this function, we use Newton's method to get a root of `f(x) := x² - a`. It involves building a
            // sequence x_n that converges toward sqrt(a). For each iteration x_n, we also define the error between
            // the current value as `ε_n = | x_n - sqrt(a) |`.
            //
            // For our first estimation, we consider `e` the smallest power of 2 which is bigger than the square root
            // of the target. (i.e. `2**(e-1) ≤ sqrt(a) < 2**e`). We know that `e ≤ 128` because `(2¹²⁸)² = 2²⁵⁶` is
            // bigger than any uint256.
            //
            // By noticing that
            // `2**(e-1) ≤ sqrt(a) < 2**e → (2**(e-1))² ≤ a < (2**e)² → 2**(2*e-2) ≤ a < 2**(2*e)`
            // we can deduce that `e - 1` is `log2(a) / 2`. We can thus compute `x_n = 2**(e-1)` using a method similar
            // to the msb function.
            uint256 aa = a;
            uint256 xn = 1;

            if (aa >= (1 << 128)) {
                aa >>= 128;
                xn <<= 64;
            }
            if (aa >= (1 << 64)) {
                aa >>= 64;
                xn <<= 32;
            }
            if (aa >= (1 << 32)) {
                aa >>= 32;
                xn <<= 16;
            }
            if (aa >= (1 << 16)) {
                aa >>= 16;
                xn <<= 8;
            }
            if (aa >= (1 << 8)) {
                aa >>= 8;
                xn <<= 4;
            }
            if (aa >= (1 << 4)) {
                aa >>= 4;
                xn <<= 2;
            }
            if (aa >= (1 << 2)) {
                xn <<= 1;
            }

            // We now have x_n such that `x_n = 2**(e-1) ≤ sqrt(a) < 2**e = 2 * x_n`. This implies ε_n ≤ 2**(e-1).
            //
            // We can refine our estimation by noticing that the middle of that interval minimizes the error.
            // If we move x_n to equal 2**(e-1) + 2**(e-2), then we reduce the error to ε_n ≤ 2**(e-2).
            // This is going to be our x_0 (and ε_0)
            xn = (3 * xn) >> 1; // ε_0 := | x_0 - sqrt(a) | ≤ 2**(e-2)

            // From here, Newton's method give us:
            // x_{n+1} = (x_n + a / x_n) / 2
            //
            // One should note that:
            // x_{n+1}² - a = ((x_n + a / x_n) / 2)² - a
            //              = ((x_n² + a) / (2 * x_n))² - a
            //              = (x_n⁴ + 2 * a * x_n² + a²) / (4 * x_n²) - a
            //              = (x_n⁴ + 2 * a * x_n² + a² - 4 * a * x_n²) / (4 * x_n²)
            //              = (x_n⁴ - 2 * a * x_n² + a²) / (4 * x_n²)
            //              = (x_n² - a)² / (2 * x_n)²
            //              = ((x_n² - a) / (2 * x_n))²
            //              ≥ 0
            // Which proves that for all n ≥ 1, sqrt(a) ≤ x_n
            //
            // This gives us the proof of quadratic convergence of the sequence:
            // ε_{n+1} = | x_{n+1} - sqrt(a) |
            //         = | (x_n + a / x_n) / 2 - sqrt(a) |
            //         = | (x_n² + a - 2*x_n*sqrt(a)) / (2 * x_n) |
            //         = | (x_n - sqrt(a))² / (2 * x_n) |
            //         = | ε_n² / (2 * x_n) |
            //         = ε_n² / | (2 * x_n) |
            //
            // For the first iteration, we have a special case where x_0 is known:
            // ε_1 = ε_0² / | (2 * x_0) |
            //     ≤ (2**(e-2))² / (2 * (2**(e-1) + 2**(e-2)))
            //     ≤ 2**(2*e-4) / (3 * 2**(e-1))
            //     ≤ 2**(e-3) / 3
            //     ≤ 2**(e-3-log2(3))
            //     ≤ 2**(e-4.5)
            //
            // For the following iterations, we use the fact that, 2**(e-1) ≤ sqrt(a) ≤ x_n:
            // ε_{n+1} = ε_n² / | (2 * x_n) |
            //         ≤ (2**(e-k))² / (2 * 2**(e-1))
            //         ≤ 2**(2*e-2*k) / 2**e
            //         ≤ 2**(e-2*k)
            xn = (xn + a / xn) >> 1; // ε_1 := | x_1 - sqrt(a) | ≤ 2**(e-4.5)  -- special case, see above
            xn = (xn + a / xn) >> 1; // ε_2 := | x_2 - sqrt(a) | ≤ 2**(e-9)    -- general case with k = 4.5
            xn = (xn + a / xn) >> 1; // ε_3 := | x_3 - sqrt(a) | ≤ 2**(e-18)   -- general case with k = 9
            xn = (xn + a / xn) >> 1; // ε_4 := | x_4 - sqrt(a) | ≤ 2**(e-36)   -- general case with k = 18
            xn = (xn + a / xn) >> 1; // ε_5 := | x_5 - sqrt(a) | ≤ 2**(e-72)   -- general case with k = 36
            xn = (xn + a / xn) >> 1; // ε_6 := | x_6 - sqrt(a) | ≤ 2**(e-144)  -- general case with k = 72

            // Because e ≤ 128 (as discussed during the first estimation phase), we know have reached a precision
            // ε_6 ≤ 2**(e-144) < 1. Given we're operating on integers, then we can ensure that xn is now either
            // sqrt(a) or sqrt(a) + 1.
            return xn - SafeCast.toUint(xn > a / xn);
        }
    }

    /**
     * @dev Calculates sqrt(a), following the selected rounding direction.
     */
    function sqrt(uint256 a, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = sqrt(a);
            return result + SafeCast.toUint(unsignedRoundsUp(rounding) && result * result < a);
        }
    }

    /**
     * @dev Return the log in base 2 of a positive value rounded towards zero.
     * Returns 0 if given 0.
     */
    function log2(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        uint256 exp;
        unchecked {
            exp = 128 * SafeCast.toUint(value > (1 << 128) - 1);
            value >>= exp;
            result += exp;

            exp = 64 * SafeCast.toUint(value > (1 << 64) - 1);
            value >>= exp;
            result += exp;

            exp = 32 * SafeCast.toUint(value > (1 << 32) - 1);
            value >>= exp;
            result += exp;

            exp = 16 * SafeCast.toUint(value > (1 << 16) - 1);
            value >>= exp;
            result += exp;

            exp = 8 * SafeCast.toUint(value > (1 << 8) - 1);
            value >>= exp;
            result += exp;

            exp = 4 * SafeCast.toUint(value > (1 << 4) - 1);
            value >>= exp;
            result += exp;

            exp = 2 * SafeCast.toUint(value > (1 << 2) - 1);
            value >>= exp;
            result += exp;

            result += SafeCast.toUint(value > 1);
        }
        return result;
    }

    /**
     * @dev Return the log in base 2, following the selected rounding direction, of a positive value.
     * Returns 0 if given 0.
     */
    function log2(uint256 value, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = log2(value);
            return result + SafeCast.toUint(unsignedRoundsUp(rounding) && 1 << result < value);
        }
    }

    /**
     * @dev Return the log in base 10 of a positive value rounded towards zero.
     * Returns 0 if given 0.
     */
    function log10(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        unchecked {
            if (value >= 10 ** 64) {
                value /= 10 ** 64;
                result += 64;
            }
            if (value >= 10 ** 32) {
                value /= 10 ** 32;
                result += 32;
            }
            if (value >= 10 ** 16) {
                value /= 10 ** 16;
                result += 16;
            }
            if (value >= 10 ** 8) {
                value /= 10 ** 8;
                result += 8;
            }
            if (value >= 10 ** 4) {
                value /= 10 ** 4;
                result += 4;
            }
            if (value >= 10 ** 2) {
                value /= 10 ** 2;
                result += 2;
            }
            if (value >= 10 ** 1) {
                result += 1;
            }
        }
        return result;
    }

    /**
     * @dev Return the log in base 10, following the selected rounding direction, of a positive value.
     * Returns 0 if given 0.
     */
    function log10(uint256 value, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = log10(value);
            return result + SafeCast.toUint(unsignedRoundsUp(rounding) && 10 ** result < value);
        }
    }

    /**
     * @dev Return the log in base 256 of a positive value rounded towards zero.
     * Returns 0 if given 0.
     *
     * Adding one to the result gives the number of pairs of hex symbols needed to represent `value` as a hex string.
     */
    function log256(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        uint256 isGt;
        unchecked {
            isGt = SafeCast.toUint(value > (1 << 128) - 1);
            value >>= isGt * 128;
            result += isGt * 16;

            isGt = SafeCast.toUint(value > (1 << 64) - 1);
            value >>= isGt * 64;
            result += isGt * 8;

            isGt = SafeCast.toUint(value > (1 << 32) - 1);
            value >>= isGt * 32;
            result += isGt * 4;

            isGt = SafeCast.toUint(value > (1 << 16) - 1);
            value >>= isGt * 16;
            result += isGt * 2;

            result += SafeCast.toUint(value > (1 << 8) - 1);
        }
        return result;
    }

    /**
     * @dev Return the log in base 256, following the selected rounding direction, of a positive value.
     * Returns 0 if given 0.
     */
    function log256(uint256 value, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = log256(value);
            return result + SafeCast.toUint(unsignedRoundsUp(rounding) && 1 << (result << 3) < value);
        }
    }

    /**
     * @dev Returns whether a provided rounding mode is considered rounding up for unsigned integers.
     */
    function unsignedRoundsUp(Rounding rounding) internal pure returns (bool) {
        return uint8(rounding) % 2 == 1;
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts-upgradeable/utils/ContextUpgradeable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.1) (utils/Context.sol)

pragma solidity ^0.8.20;
import {Initializable} from "../proxy/utils/Initializable.sol";

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
abstract contract ContextUpgradeable is Initializable {
    function __Context_init() internal onlyInitializing {
    }

    function __Context_init_unchained() internal onlyInitializing {
    }
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


// ===== FILE: node_modules/_openzeppelin/contracts/interfaces/draft-IERC6093.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (interfaces/draft-IERC6093.sol)
pragma solidity ^0.8.20;

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


// ===== FILE: node_modules/_openzeppelin/contracts/utils/Errors.sol =====
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


// ===== FILE: node_modules/_openzeppelin/contracts/utils/Context.sol =====
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


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/interface/IAllocateModule.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

/**
 * @title IAllocateModule
 * @dev Interface for the AllocateModule contract that handles fund allocation and deallocation across multiple strategies.
 * @dev This module enables the vault to efficiently manage funds across different yield-generating strategies
 *      by batching multiple allocation/deallocation operations in a single transaction.
 *
 * @notice The AllocateModule serves as a coordinator for strategy operations, allowing the vault to:
 * - Allocate funds to multiple strategies in a single call
 * - Deallocate funds from multiple strategies in a single call
 * - Maintain accurate accounting of allocated amounts per strategy
 *
 * @notice This module is typically used during rebalancing operations where the vault needs to
 *         adjust allocations across multiple strategies based on current market conditions,
 *         strategy performance, or allocation targets.
 */
interface IAllocateModule {
    /**
     * @dev Emitted when funds are allocated or deallocated from a strategy.
     *
     * @param strategy The address of the strategy contract.
     * @param isDeposit True if this is an allocation (deposit) operation, false if it's a deallocation (withdrawal).
     * @param amount The amount of funds allocated or deallocated.
     * @param extraData Arbitrary calldata passed to the strategy's allocateFunds or deallocateFunds function.
     */
    event AllocatedFunds(address indexed strategy, bool indexed isDeposit, uint256 amount, bytes extraData);

    /**
     * @dev Structure containing parameters for a single allocation or deallocation operation.
     *
     * @param isDeposit True if this is an allocation (deposit) operation, false if it's a deallocation (withdrawal).
     * @param strategy The address of the strategy contract to interact with.
     * @param extraData Arbitrary calldata to pass to the strategy's allocateFunds or deallocateFunds function.
     *                  This allows for strategy-specific parameters like slippage tolerance, routing info, etc.
     */
    struct AllocateParams {
        bool isDeposit;
        address strategy;
        bytes extraData;
    }

    /**
     * @dev Executes multiple allocation and deallocation operations across different strategies in a single transaction.
     * @dev This function processes an array of allocation parameters, calling the appropriate strategy functions
     *      and updating the vault's internal accounting for each strategy.
     *
     * @param data ABI-encoded array of AllocateParams structures containing the operations to execute.
     *             Each AllocateParams specifies whether to allocate or deallocate funds, which strategy to use,
     *             and any additional data required by the strategy.
     *
     * @notice The function iterates through all provided parameters and:
     * - For deposits (isDeposit = true): Calls strategy.allocateFunds() and increases allocated amount
     * - For withdrawals (isDeposit = false): Calls strategy.deallocateFunds() and decreases allocated amount
     *
     * @notice All operations are executed atomically - if any single operation fails, the entire transaction reverts.
     *
     * @notice The function updates the vault's internal strategy accounting to track the total amount
     *         allocated to each strategy, which is used for yield calculation and strategy limits.
     */
    function allocateFunds(bytes calldata data) external;
}


// ===== FILE: node_modules/_openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (access/Ownable.sol)

pragma solidity ^0.8.20;

import {ContextUpgradeable} from "../utils/ContextUpgradeable.sol";
import {Initializable} from "../proxy/utils/Initializable.sol";

/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * The initial owner is set to the address provided by the deployer. This can
 * later be changed with {transferOwnership}.
 *
 * This module is used through inheritance. It will make available the modifier
 * `onlyOwner`, which can be applied to your functions to restrict their use to
 * the owner.
 */
abstract contract OwnableUpgradeable is Initializable, ContextUpgradeable {
    /// @custom:storage-location erc7201:openzeppelin.storage.Ownable
    struct OwnableStorage {
        address _owner;
    }

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.Ownable")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant OwnableStorageLocation = 0x9016d09d72d40fdae2fd8ceac6b6234c7706214fd39c1cd1e609a0528c199300;

    function _getOwnableStorage() private pure returns (OwnableStorage storage $) {
        assembly {
            $.slot := OwnableStorageLocation
        }
    }

    /**
     * @dev The caller account is not authorized to perform an operation.
     */
    error OwnableUnauthorizedAccount(address account);

    /**
     * @dev The owner is not a valid owner account. (eg. `address(0)`)
     */
    error OwnableInvalidOwner(address owner);

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev Initializes the contract setting the address provided by the deployer as the initial owner.
     */
    function __Ownable_init(address initialOwner) internal onlyInitializing {
        __Ownable_init_unchained(initialOwner);
    }

    function __Ownable_init_unchained(address initialOwner) internal onlyInitializing {
        if (initialOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(initialOwner);
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    /**
     * @dev Returns the address of the current owner.
     */
    function owner() public view virtual returns (address) {
        OwnableStorage storage $ = _getOwnableStorage();
        return $._owner;
    }

    /**
     * @dev Throws if the sender is not the owner.
     */
    function _checkOwner() internal view virtual {
        if (owner() != _msgSender()) {
            revert OwnableUnauthorizedAccount(_msgSender());
        }
    }

    /**
     * @dev Leaves the contract without owner. It will not be possible to call
     * `onlyOwner` functions. Can only be called by the current owner.
     *
     * NOTE: Renouncing ownership will leave the contract without an owner,
     * thereby disabling any functionality that is only available to the owner.
     */
    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(newOwner);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Internal function without access restriction.
     */
    function _transferOwnership(address newOwner) internal virtual {
        OwnableStorage storage $ = _getOwnableStorage();
        address oldOwner = $._owner;
        $._owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts/utils/ReentrancyGuardTransient.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/ReentrancyGuardTransient.sol)

pragma solidity ^0.8.24;

import {TransientSlot} from "./TransientSlot.sol";

/**
 * @dev Variant of {ReentrancyGuard} that uses transient storage.
 *
 * NOTE: This variant only works on networks where EIP-1153 is available.
 *
 * _Available since v5.1._
 */
abstract contract ReentrancyGuardTransient {
    using TransientSlot for *;

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.ReentrancyGuard")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant REENTRANCY_GUARD_STORAGE =
        0x9b779b17422d0df92223018b32b4d1fa46e071723d6817e2486d003becc55f00;

    /**
     * @dev Unauthorized reentrant call.
     */
    error ReentrancyGuardReentrantCall();

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
        if (_reentrancyGuardEntered()) {
            revert ReentrancyGuardReentrantCall();
        }

        // Any calls to nonReentrant after this point will fail
        REENTRANCY_GUARD_STORAGE.asBoolean().tstore(true);
    }

    function _nonReentrantAfter() private {
        REENTRANCY_GUARD_STORAGE.asBoolean().tstore(false);
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        return REENTRANCY_GUARD_STORAGE.asBoolean().tload();
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/AccessControlLib.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {AccessControlUpgradeable} from "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import {
    AccessControlEnumerableUpgradeable
} from "@openzeppelin/contracts-upgradeable/access/extensions/AccessControlEnumerableUpgradeable.sol";
import {IAccessControl} from "@openzeppelin/contracts/access/IAccessControl.sol";
import {EnumerableSet} from "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";

library ACL_OZ_5_2_0_Lib {
    using EnumerableSet for EnumerableSet.AddressSet;
    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.AccessControl")) - 1)) & ~bytes32(uint256(0xff))

    bytes32 private constant AccessControlStorageLocation =
        0x02dd7bc7dec4dceedda775e58dd541e08a116c6c53815c0bd028192f7b626800;

    function __getAccessControlStorage()
        private
        pure
        returns (AccessControlUpgradeable.AccessControlStorage storage $)
    {
        assembly {
            $.slot := AccessControlStorageLocation
        }
    }

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.AccessControlEnumerable")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant AccessControlEnumerableStorageLocation =
        0xc1f6fe24621ce81ec5827caf0253cadb74709b061630e6b55e82371705932000;

    function __getAccessControlEnumerableStorage()
        private
        pure
        returns (AccessControlEnumerableUpgradeable.AccessControlEnumerableStorage storage $)
    {
        assembly {
            $.slot := AccessControlEnumerableStorageLocation
        }
    }

    /**
     * @dev Overload {AccessControl-_grantRole} to track enumerable memberships
     */
    function _grantRole(bytes32 role, address account, address sender) internal returns (bool) {
        AccessControlEnumerableUpgradeable.AccessControlEnumerableStorage storage $ace =
            __getAccessControlEnumerableStorage();
        bool granted = __grantRole(role, account, sender);
        if (granted) {
            $ace._roleMembers[role].add(account);
        }
        return granted;
    }

    function __grantRole(bytes32 role, address account, address sender) private returns (bool) {
        AccessControlUpgradeable.AccessControlStorage storage $ac = __getAccessControlStorage();
        if (!__hasRole(role, account)) {
            $ac._roles[role].hasRole[account] = true;
            emit IAccessControl.RoleGranted(role, account, sender);
            return true;
        } else {
            return false;
        }
    }

    /**
     * @dev Returns `true` if `account` has been granted `role`.
     */
    function __hasRole(bytes32 role, address account) private view returns (bool) {
        AccessControlUpgradeable.AccessControlStorage storage $ac = __getAccessControlStorage();
        return $ac._roles[role].hasRole[account];
    }

    /**
     * @dev Returns the admin role that controls `role`. See {grantRole} and
     * {revokeRole}.
     *
     * To change a role's admin, use {_setRoleAdmin}.
     */
    function __getRoleAdmin(bytes32 role) private view returns (bytes32) {
        AccessControlUpgradeable.AccessControlStorage storage $ac = __getAccessControlStorage();
        return $ac._roles[role].adminRole;
    }

    /**
     * @dev Sets `adminRole` as ``role``'s admin role.
     *
     * Emits a {RoleAdminChanged} event.
     */
    function _setRoleAdmin(bytes32 role, bytes32 adminRole) internal {
        AccessControlUpgradeable.AccessControlStorage storage $ac = __getAccessControlStorage();
        bytes32 previousAdminRole = __getRoleAdmin(role);
        $ac._roles[role].adminRole = adminRole;
        emit IAccessControl.RoleAdminChanged(role, previousAdminRole, adminRole);
    }

    /**
     * @dev Returns the number of accounts that have `role`. Can be used
     * together with {getRoleMember} to enumerate all bearers of a role.
     */
    function _getRoleMemberCount(bytes32 role) internal view returns (uint256) {
        AccessControlEnumerableUpgradeable.AccessControlEnumerableStorage storage $ace =
            __getAccessControlEnumerableStorage();
        return $ace._roleMembers[role].length();
    }

    /**
     * @dev Returns one of the accounts that have `role`. `index` must be a
     * value between 0 and {getRoleMemberCount}, non-inclusive.
     */
    function _getRoleMember(bytes32 role, uint256 index) internal view returns (address) {
        AccessControlEnumerableUpgradeable.AccessControlEnumerableStorage storage $ace =
            __getAccessControlEnumerableStorage();
        return $ace._roleMembers[role].at(index);
    }

    /**
     * @dev Revokes `role` from `account`.
     *
     * Internal function without access restriction.
     *
     * Emits a {RoleRevoked} event.
     */
    function _revokeRole(bytes32 role, address account, address sender) internal returns (bool) {
        AccessControlEnumerableUpgradeable.AccessControlEnumerableStorage storage $ace =
            __getAccessControlEnumerableStorage();
        bool revoked = __revokeRole(role, account, sender);
        if (revoked) {
            $ace._roleMembers[role].remove(account);
        }
        return revoked;
    }

    function __revokeRole(bytes32 role, address account, address sender) private returns (bool) {
        AccessControlUpgradeable.AccessControlStorage storage $ac = __getAccessControlStorage();
        if (__hasRole(role, account)) {
            $ac._roles[role].hasRole[account] = false;
            emit IAccessControl.RoleRevoked(role, account, sender);
            return true;
        } else {
            return false;
        }
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/storage/ConcreteAsyncVaultImplStorageLib.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

library ConcreteAsyncVaultImplStorageLib {
    /// @dev keccak256(abi.encode(uint256(keccak256("concrete.storage.ConcreteAsyncVaultImplStorage")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant ConcreteAsyncVaultImplStorageLocation =
        0x0ada5b606f7944319310c49c0f9f30d6272793a991bd2b9c3db8049867746700;

    /// @custom:storage-location erc7201:concrete.storage.ConcreteAsyncVaultImplStorage
    struct ConcreteAsyncVaultImplStorage {
        // Current epoch ID for async withdrawals
        uint256 latestEpochID;
        // Assets available for past withdrawals (denominated in underlying asset)
        uint256 pastEpochsUnclaimedAssets;
        // Mapping from epoch ID to total shares requested in that epoch
        mapping(uint256 => uint256) totalRequestedSharesPerEpoch;
        // Mapping from user address to epoch ID to shares requested
        mapping(address user => mapping(uint256 epochID => uint256 shares)) userEpochRequests;
        // Mapping from epoch ID to share price when that epoch was processed
        mapping(uint256 => uint256) epochPricePerSharePlusOne;
        // Whether the queue is active
        bool isQueueActive;
        // Unwind cost cap for priority withdrawals in basis points (10000 = 100%)
        uint16 unwindCostCapBP;
    }

    /**
     * @dev Returns the storage struct for ConcreteAsyncVaultImpl
     */
    function fetch() internal pure returns (ConcreteAsyncVaultImplStorage storage $) {
        assembly {
            $.slot := ConcreteAsyncVaultImplStorageLocation
        }
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/interface/IConcreteAsyncVaultImpl.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import {IConcreteStandardVaultImpl} from "./IConcreteStandardVaultImpl.sol";

/**
 * @title IConcreteAsyncVaultImpl
 * @dev Interface for the async vault implementation that provides epoch-based withdrawal management.
 * @dev This interface extends the standard vault functionality with asynchronous withdrawal capabilities,
 * allowing for better liquidity management and batch processing of withdrawal requests.
 * @dev Users submit withdrawal requests that are queued in epochs, processed by allocators when liquidity
 * is available, and then claimed by users after processing is complete.
 */
interface IConcreteAsyncVaultImpl is IConcreteStandardVaultImpl {
    enum EpochState {
        // Epoch is not active
        Inactive,
        // An epoch is active if it is receiving requests and is not closed
        Active,
        // An epoch is processing if it is closed (cannot receive requests anymore) and has not been processed
        Processing,
        // An epoch is processed if it has been processed and has a price locked
        Processed
    }

    /**
     * @dev Thrown when attempting to operate on a zero address.
     */
    error ZeroAddress();

    /**
     * @dev Thrown when attempting to withdraw zero shares.
     */
    error ZeroShares();

    /**
     * @dev Thrown when attempting to operate on an epoch with no requesting shares.
     */
    error NoRequestingShares();

    /**
     * @dev Thrown when attempting to claim a request that is not claimable.
     */
    error NoClaimableRequest();

    /**
     * @dev Thrown when there are no redeemable assets available.
     */
    error NoRedeemableAssets();

    /**
     * @dev Thrown when attempting to roll to next epoch while current epoch is not processed.
     */
    error EpochNotProcessed(uint256 epochID);

    /**
     * @dev Thrown when attempting to cancel a request for an epoch that has already been processed.
     */
    error EpochAlreadyProcessed(uint256 epochID);

    /**
     * @dev Thrown when attempting to claim with empty epoch IDs.
     */
    error EmptyEpochIDs();

    /**
     * @dev Thrown when attempting to claim with empty users.
     */
    error EmptyUsers();

    /**
     * @dev Thrown when attempting to claim with unwind cost that exceeds the unwind cost cap.
     */
    error UnwindCostTooHigh();

    /**
     * @dev Thrown when attempting to set an invalid unwind cost cap.
     */
    error InvalidUnwindCostCap();

    /**
     * @dev Emitted when the withdrawal queue is initialized.
     * @param epochID The initial epoch ID.
     */
    event WithdrawalQueueInitialized(uint256 epochID);

    /**
     * @dev Thrown when attempting to close an epoch that and previous epoch was not processed.
     */
    error PreviousEpochNotProcessed(uint256 epochID);

    /**
     * @dev Thrown when attempting to process an epoch that was already processed.
     */
    error PreviousEpochAlreadyProcessed(uint256 epochID);

    /**
     * @dev Thrown when attempting to process an epoch that is already closed.
     * @param epochID The epoch that was already closed.
     */
    error EpochAlreadyClosed(uint256 epochID);

    /**
     * @dev Thrown when the provided epoch ID does not match the expected epoch.
     * @param providedEpochID The epoch ID that was provided.
     * @param expectedEpochID The epoch ID that was expected.
     */
    error EpochMismatch(uint256 providedEpochID, uint256 expectedEpochID);

    /**
     * @dev Emitted when a user submits a withdrawal request that is queued for epoch processing.
     * @param owner The address of the user making the request.
     * @param assets The amount of assets requested for withdrawal.
     * @param shares The amount of shares transferred to the vault for the request.
     * @param epochID The epoch in which the request was queued.
     */
    event QueuedWithdrawal(
        address indexed caller,
        address indexed receiver,
        address indexed owner,
        uint256 assets,
        uint256 shares,
        uint256 epochID
    );

    /**
     * @dev Emitted when a user cancels their pending withdrawal request.
     * @param owner The address of the user cancelling the request.
     * @param shares The amount of shares returned to the user.
     * @param epochID The epoch from which the request was cancelled.
     */
    event RequestCancelled(address indexed owner, uint256 shares, uint256 epochID);

    /**
     * @dev Emitted when a user claims their processed withdrawal request.
     * @param owner The address of the user claiming the withdrawal.
     * @param assets The amount of assets transferred to the user.
     * @param epochIDs The epoch IDs from which the request was claimed.
     */
    event RequestClaimed(address indexed owner, uint256 assets, uint256[] epochIDs);

    /**
     * @dev Emitted when an epoch's withdrawal requests are processed.
     * @param epochID The epoch ID that was processed.
     * @param shares The total shares processed in the epoch.
     * @param assets The total assets reserved for the epoch.
     * @param sharePrice The share price locked for the epoch.
     */
    event EpochProcessed(uint256 epochID, uint256 shares, uint256 assets, uint256 sharePrice);

    /**
     * @dev Emitted when a user's request is moved to the next epoch.
     * @param user The address of the user whose request was moved.
     * @param shares The amount of shares moved.
     * @param currentEpochID The epoch from which the request was moved.
     * @param nextEpochID The epoch to which the request was moved.
     */
    event RequestMovedToNextEpoch(address indexed user, uint256 shares, uint256 currentEpochID, uint256 nextEpochID);

    /**
     * @dev Emitted when a user's priority withdrawal is claimed.
     * @param user The address of the user claiming the withdrawal.
     * @param shares The amount of shares claimed/fulfilled.
     * @param grossAssets The amount of gross assets before deducting unwind cost based on current exchange rate
     * @param unwindCost The amount paid by user as a priority wwl fee (operational cost).
     * @param netAssets The amount of assets claimed by user after deducting unwind cost.
     * @param epochID The epoch from which the withdrawal was claimed.
     */
    event PriorityWithdrawalClaimed(
        address indexed user,
        uint256 shares,
        uint256 grossAssets,
        uint256 unwindCost,
        uint256 netAssets,
        uint256 epochID
    );

    /**
     * @dev Emitted when an individual withdrawal request is fulfilled from a closed, unprocessed epoch.
     * @param user The address of the user whose request was fulfilled.
     * @param epochID The closed epoch from which the request was fulfilled.
     * @param shares Number of shares burned for this fulfilment.
     * @param assets Amount of underlying tokens transferred to the user.
     */
    event PartialEpochRequestProcessed(address indexed user, uint256 epochID, uint256 shares, uint256 assets);

    /**
     * @dev Emitted when an epoch is closed.
     * @param epochID The epoch that was closed.
     */
    event EpochClosed(uint256 epochID);

    /**
     * @dev Emitted when the queue is toggled.
     * @param isQueueActive The active status of the queue.
     */
    event QueueActiveToggled(bool isQueueActive);

    /**
     * @dev This struct definition is maintained for interface compatibility.
     */
    struct RedeemRequest {
        uint256 requestEpoch;
        uint256 requestShares;
    }

    /**
     * @notice Cancel pending redeem request for a specific epoch (only unprocessed epochs)
     * @dev Users can only cancel requests from epochs that haven't been processed yet (no price locked)
     * @dev Returns shares back to the user and updates epoch accounting
     * @param user The user address to cancel the request for
     * @param epochID The epoch ID from which to cancel the request
     */
    function cancelRequest(address user, uint256 epochID) external;

    /**
     * @notice Cancel pending redeem request for a specific epoch (only unprocessed epochs)
     * @dev Users can only cancel requests from epochs that haven't been processed yet (no price locked)
     * @dev Returns shares back to the user and updates epoch accounting
     * @param epochID The epoch ID from which to cancel the request
     */
    function cancelRequest(uint256 epochID) external;

    /**
     * @notice Claim processed redeem requests from specified epochs
     * @dev Processes each epoch internally, aggregates assets and shares, then burns and transfers once
     * @dev More gas efficient than claiming epochs individually
     * @dev Skips epochs with no claimable amounts (zero shares or unprocessed epochs)
     * @param epochIDs Array of epoch IDs to claim from
     */
    function claimWithdrawal(uint256[] calldata epochIDs) external;

    /**
     * @notice Claim processed redeem requests from specified epochs
     * @dev Processes each epoch internally, aggregates assets and shares, then burns and transfers once
     * @dev More gas efficient than claiming epochs individually
     * @dev Skips epochs with no claimable amounts (zero shares or unprocessed epochs)
     * @param user The user address to claim from
     * @param epochIDs Array of epoch IDs to claim from
     */
    function claimWithdrawal(address user, uint256[] calldata epochIDs) external;

    /**
     * @notice Claim a priority withdrawal.
     * @dev Only callable by PRIORITY_WITHDRAWAL_EXECUTOR role.
     *      Burns the user's queued shares, transfers `grossAssets - unwindCost` to the
     *      user, deducts the net amount from `cachedTotalAssets`, and calls `IAsyncAccounting.adjustTotalAssets`
     *      on the strategy to record the unwind cost.  The next yield accrual will
     *      reconcile the vault's tracked allocation with the strategy.
     * @param user The user address to claim from.
     * @param sharesToFulfil The number of shares to burn.
     * @param unwindCost The cost charged for early unwinding.
     * @param strategy The strategy whose `totalAllocatedValue` will be reduced by the
     *                 unwind cost. Must not be `address(0)`.
     */
    function claimPriorityWithdrawal(address user, uint256 sharesToFulfil, uint256 unwindCost, address strategy)
        external;

    /**
     * @notice Close the current epoch
     * @dev Only callable by WITHDRAWAL_MANAGER role
     * @dev Closes the current epoch and increments the epoch ID
     * @dev Can only be called if the previous epoch is processed (!)
     */
    function closeEpoch() external;

    /**
     * @notice Process all pending redeem requests for a specific epoch
     * @dev Harvests all strategies to get current accurate pricing before processing
     * @dev Calculates share price and reserves required assets for the epoch
     * @dev Can process any epoch with pending requests, enabling historical processing
     * @dev Only callable by WITHDRAWAL_MANAGER role
     */
    function processEpoch() external;

    /**
     * @notice Fulfil all or part of a user's withdrawal request from a closed,
     *         unprocessed epoch.
     * @dev Burns the specified shares at the current exchange rate and transfers the
     *      corresponding assets immediately (no separate claim step).  The epoch is
     *      NOT marked as processed; `processEpoch()` must still be called to
     *      finalise the epoch.
     * @dev Only callable by WITHDRAWAL_MANAGER role.
     * @param user The user whose closed-epoch request to fulfil.
     * @param epochID The closed epoch to target. Must be closed and not yet processed
     *                (intent-validation parameter).
     * @param amount Number of shares to process. Pass `type(uint256).max` to process
     *               the user's entire request.
     */
    function processPartialEpochRequest(address user, uint256 epochID, uint256 amount) external;

    /**
     * @notice Move a user's withdrawal request from the closed, unprocessed epoch
     *         (`latestEpochID - 1`) to the current active epoch (`latestEpochID`).
     * @dev Reverts with `EpochMismatch` if `fromEpochID` does not match
     *      `latestEpochID - 1`, or with `EpochAlreadyProcessed` if the
     *      previous epoch is already processed.
     * @dev Only callable by WITHDRAWAL_MANAGER role.
     * @param user The user address whose request to move.
     * @param fromEpochID The closed epoch to move from. Must equal `latestEpochID - 1`
     *                    (intent-validation parameter).
     * @param amount Number of shares to move. Pass `type(uint256).max` to move the
     *               user's entire request.
     */
    function moveRequestToNextEpoch(address user, uint256 fromEpochID, uint256 amount) external;

    /**
     * @notice Get claimable redeem request from a specific epoch
     * @dev Returns the amount of assets claimable by the caller from the specified epoch
     * @dev Returns 0 if epoch is not processed, no request exists, or epoch is current/future
     * @param epochID The epoch ID to check
     * @return assets The amount of assets claimable from the epoch
     */
    function getUserEpochRequestInAssets(address user, uint256 epochID) external view returns (uint256 assets);

    /**
     * @notice Get user's redeem request for a specific epoch
     * @dev Returns the amount of shares requested by the specified user in the epoch
     * @param user The user address to check
     * @param epochID The epoch ID to check
     * @return shares The amount of shares requested by the user in the epoch
     */
    function getUserEpochRequest(address user, uint256 epochID) external view returns (uint256 shares);

    /**
     * @notice Get the state of an epoch
     * @dev Returns the state of an epoch
     * @param epochID The epoch ID to check
     * @return The state of the epoch
     */
    function getEpochState(uint256 epochID) external view returns (EpochState);

    /**
     * @notice Get current epoch ID
     * @dev New withdrawal requests are queued in this epoch
     * @return The current epoch ID
     */
    function latestEpochID() external view returns (uint256);

    /**
     * @notice Get total redeemable assets from past epochs
     * @dev This amount is subtracted from totalAssets() to exclude reserved funds from active vault operations
     * @return The total assets reserved for past epoch claims
     */
    function pastEpochsUnclaimedAssets() external view returns (uint256);

    /**
     * @notice Get total requested shares in a specific epoch
     * @dev Returns the total shares from all users that requested withdrawal in the epoch
     * @dev Returns 0 after epoch is processed (shares moved to requestingSharesInPast)
     * @param epochID The epoch ID to check
     * @return The total shares requested in the epoch
     */
    function totalRequestedSharesPerEpoch(uint256 epochID) external view returns (uint256);

    /**
     * @notice Get total requested shares in the active and processing epochs
     * @dev Returns the total shares from all users that requested withdrawal in the active and processing epochs
     * @return activeShares The total shares requested in the active epoch
     * @return processingShares The total shares requested in the processing epoch
     * @return processedShares The total shares requested in the last processed epoch
     */
    function totalRequestedSharesForCurrentEpochs()
        external
        view
        returns (uint256 activeShares, uint256 processingShares, uint256 processedShares);

    /**
     * @notice Get share price for a specific epoch
     * @dev Returns the share price that was locked when the epoch was processed
     * @dev Returns 0 if the epoch has not been processed yet
     * @param epochID The epoch ID to check
     * @return The share price locked when the epoch was processed (in assets per 1e18 shares)
     */
    function getEpochPricePerShare(uint256 epochID) external view returns (uint256);

    /**
     * @notice Get whether the queue is active
     * @dev Returns true if the queue is active, false otherwise
     * @return The active status of the queue
     */
    function isQueueActive() external view returns (bool);

    /**
     * @notice Toggle the active status of the queue
     * @dev Only callable by the ALLOCATOR role
     */
    function toggleQueueActive() external;

    /**
     * @notice Get the unwind cost cap for priority withdrawals
     * @dev Returns the unwind cost cap in basis points (10000 = 100%)
     * @return The unwind cost cap in basis points
     */
    function getUnwindCostCap() external view returns (uint16);

    /**
     * @notice Set the unwind cost cap for priority withdrawals
     * @dev Only callable by the VAULT_MANAGER role
     * @dev Unwind cost cap is in basis points (10000 = 100%)
     * @param unwindCostCap_ The new unwind cost cap in basis points
     */
    function setUnwindCostCap(uint16 unwindCostCap_) external;
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/Time.sol =====
// SPDX-License-Identifier: UNLICENSED
// OpenZeppelin Contracts (last updated v5.1.0) (utils/types/Time.sol)
pragma solidity ^0.8.24;

import {SafeCast} from "@openzeppelin-contracts/utils/math/SafeCast.sol";

/**
 * @dev This library provides helpers for manipulating time-related objects.
 *
 * It uses the following types:
 * - `uint48` for timepoints
 *
 */
library Time {
    /**
     * @dev Get the block timestamp as a Timepoint.
     */
    function timestamp() internal view returns (uint32) {
        return SafeCast.toUint32(block.timestamp);
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/Constants.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

library ConcreteV2ConstantsLib {
    /// @dev Fee denominator for basis points calculation (10_000 = 100%)
    uint16 public constant BASIS_POINTS_DENOMINATOR = 10_000;
}

library ConcreteV2FeeParamsLib {
    /// @dev Maximum management fee in basis points (10% = 1,000 bps)
    uint16 public constant MAX_MANAGEMENT_FEE = 1_000;

    /// @dev Maximum performance fee in basis points (100% = 10,000 bps)
    uint16 public constant MAX_PERFORMANCE_FEE = 10_000;
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/AsyncVaultHelperLib.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import {ConcreteV2ConversionLib as ConversionLib} from "../lib/Conversion.sol";
import {Math} from "@openzeppelin-contracts/utils/math/Math.sol";
import {SafeERC20} from "@openzeppelin-contracts/token/ERC20/utils/SafeERC20.sol";
import {ERC20_OZ_5_2_0_Lib} from "../lib/ERC20Lib.sol";
import {IERC20} from "@openzeppelin-contracts/token/ERC20/IERC20.sol";
import {ConcreteAsyncVaultImplStorageLib} from "../lib/storage/ConcreteAsyncVaultImplStorageLib.sol";
import {IConcreteAsyncVaultImpl} from "../interface/IConcreteAsyncVaultImpl.sol";
import {IConcreteStandardVaultImpl} from "../interface/IConcreteStandardVaultImpl.sol";
import {
    ConcreteCachedVaultStateStorageLib as CachedVaultStateLib
} from "../lib/storage/ConcreteCachedVaultStateStorageLib.sol";
import {ConcreteStandardVaultImplStorageLib as SVLib} from "../lib/storage/ConcreteStandardVaultImplStorageLib.sol";
import {EnumerableSet} from "@openzeppelin-contracts/utils/structs/EnumerableSet.sol";
import {IAsyncAccounting} from "../interface/IAsyncAccounting.sol";

struct ClaimWithdrawalParams {
    uint256 epochID;
    uint256 latestEpochID;
    uint256 epochPrice;
    bool isProcessed;
    uint256 totalAssetsSum;
}

library AsyncVaultHelperLib {
    using Math for uint256;
    using SafeERC20 for IERC20;

    /// @dev Basis points constant (10000 = 100%)
    uint256 public constant BASIS_POINTS = 10000;

    // ─────────────────────────────────────────────────────────────────────────────
    // Epoch Lifecycle Functions
    // ─────────────────────────────────────────────────────────────────────────────

    function closeEpoch() public {
        ConcreteAsyncVaultImplStorageLib.ConcreteAsyncVaultImplStorage storage $ =
            ConcreteAsyncVaultImplStorageLib.fetch();

        uint256 currentEpoch = $.latestEpochID;

        // Check if previous epoch is processed or if this is the first epoch
        require(
            currentEpoch == 1 || _epochProcessed(currentEpoch - 1),
            IConcreteAsyncVaultImpl.PreviousEpochNotProcessed(currentEpoch - 1)
        );

        // Increment to next epoch (safe to use unchecked as overflow is unrealistic)
        $.latestEpochID = currentEpoch + 1;

        emit IConcreteAsyncVaultImpl.EpochClosed(currentEpoch);
    }

    function processEpoch(uint256 sharePrice, uint256 availableAssets, uint8 decimals) public {
        ConcreteAsyncVaultImplStorageLib.ConcreteAsyncVaultImplStorage storage $ =
            ConcreteAsyncVaultImplStorageLib.fetch();

        uint256 previousEpochID = $.latestEpochID - 1;
        // previous epoch should be not be processed
        require(
            !_epochProcessed(previousEpochID), IConcreteAsyncVaultImpl.PreviousEpochAlreadyProcessed(previousEpochID)
        );
        uint256 requestingShares = $.totalRequestedSharesPerEpoch[previousEpochID];
        uint256 assetsNeeded = 0;
        // add 1 to indicate that the epoch is processed (price cannot be 0)
        $.epochPricePerSharePlusOne[previousEpochID] = sharePrice + 1;

        if (requestingShares == 0) {
            require(availableAssets >= $.pastEpochsUnclaimedAssets, IConcreteStandardVaultImpl.InsufficientBalance());
        } else {
            assetsNeeded = requestingShares.mulDiv(sharePrice, 10 ** decimals, Math.Rounding.Floor);

            require(
                availableAssets >= $.pastEpochsUnclaimedAssets + assetsNeeded,
                IConcreteStandardVaultImpl.InsufficientBalance()
            );
            ERC20_OZ_5_2_0_Lib._burn(address(this), requestingShares);

            // Update accounting
            $.pastEpochsUnclaimedAssets += assetsNeeded;
            CachedVaultStateLib.fetch().cachedTotalAssets -= assetsNeeded;
        }

        emit IConcreteAsyncVaultImpl.EpochProcessed(previousEpochID, requestingShares, assetsNeeded, sharePrice);
    }

    function processPartialEpochRequest(
        address asset,
        address user,
        uint256 epochID,
        uint256 amount,
        uint256 sharePrice,
        uint256 availableAssets,
        uint8 decimals
    ) public {
        require(amount > 0, IConcreteAsyncVaultImpl.ZeroShares());

        ConcreteAsyncVaultImplStorageLib.ConcreteAsyncVaultImplStorage storage $ =
            ConcreteAsyncVaultImplStorageLib.fetch();
        CachedVaultStateLib.ConcreteCachedVaultStateStorage storage $cached = CachedVaultStateLib.fetch();

        uint256 closedEpochID = $.latestEpochID - 1; // closed epoch is the previous epoch
        require(epochID == closedEpochID, IConcreteAsyncVaultImpl.EpochMismatch(epochID, closedEpochID));
        require(!_epochProcessed(closedEpochID), IConcreteAsyncVaultImpl.EpochAlreadyProcessed(closedEpochID));
        require(user != address(0), IConcreteAsyncVaultImpl.ZeroAddress());

        uint256 userRequestedShares = $.userEpochRequests[user][epochID];
        require(userRequestedShares > 0, IConcreteAsyncVaultImpl.NoRequestingShares());

        uint256 shares = amount == type(uint256).max ? userRequestedShares : amount;
        require(shares <= userRequestedShares, IConcreteStandardVaultImpl.InsufficientShares());

        uint256 assets = shares.mulDiv(sharePrice, 10 ** decimals, Math.Rounding.Floor);
        require(assets > 0, IConcreteStandardVaultImpl.InsufficientAssets());

        require(
            availableAssets >= $.pastEpochsUnclaimedAssets + assets, IConcreteStandardVaultImpl.InsufficientBalance()
        );

        $.userEpochRequests[user][epochID] -= shares;
        $.totalRequestedSharesPerEpoch[epochID] -= shares;

        $cached.cachedTotalAssets -= assets;

        ERC20_OZ_5_2_0_Lib._burn(address(this), shares);

        IERC20(asset).safeTransfer(user, assets);

        emit IConcreteAsyncVaultImpl.PartialEpochRequestProcessed(user, epochID, shares, assets);
    }

    function claimWithdrawal(address asset, address user, uint256[] calldata epochIDs, uint8 decimals) public {
        claimWithdrawalTo(asset, user, epochIDs, decimals, user);
    }

    function claimWithdrawalTo(
        address asset,
        address user,
        uint256[] calldata epochIDs,
        uint8 decimals,
        address recipient
    ) public {
        ConcreteAsyncVaultImplStorageLib.ConcreteAsyncVaultImplStorage storage $ =
            ConcreteAsyncVaultImplStorageLib.fetch();

        ClaimWithdrawalParams memory wwl;
        wwl.latestEpochID = $.latestEpochID;

        uint256 len = epochIDs.length;
        for (uint256 i = 0; i < len; i++) {
            wwl.epochID = epochIDs[i];
            (wwl.epochPrice, wwl.isProcessed) = _epochPriceAndProcessedFlag(wwl.epochID);
            if (!wwl.isProcessed) revert IConcreteAsyncVaultImpl.EpochNotProcessed(wwl.epochID);
            wwl.totalAssetsSum += getUserEpochRequestInAssets(
                user, wwl.epochID, wwl.latestEpochID, wwl.epochPrice, decimals
            );
            $.userEpochRequests[user][wwl.epochID] = 0;
        }

        require(wwl.totalAssetsSum > 0, IConcreteAsyncVaultImpl.NoClaimableRequest());

        // Update accounting
        $.pastEpochsUnclaimedAssets -= wwl.totalAssetsSum;

        require(recipient != address(0), IConcreteAsyncVaultImpl.ZeroAddress());

        // Transfer all assets at once
        IERC20(asset).safeTransfer(recipient, wwl.totalAssetsSum);

        emit IConcreteAsyncVaultImpl.RequestClaimed(user, wwl.totalAssetsSum, epochIDs);
    }

    function claimUsersBatch(address asset, address[] calldata users, uint256 epochID, uint8 decimals) public {
        ConcreteAsyncVaultImplStorageLib.ConcreteAsyncVaultImplStorage storage $ =
            ConcreteAsyncVaultImplStorageLib.fetch();

        ClaimWithdrawalParams memory wwl;
        wwl.latestEpochID = $.latestEpochID;
        wwl.epochID = epochID;

        (wwl.epochPrice, wwl.isProcessed) = _epochPriceAndProcessedFlag(wwl.epochID);
        if (!wwl.isProcessed) revert IConcreteAsyncVaultImpl.EpochNotProcessed(wwl.epochID);

        wwl.latestEpochID = $.latestEpochID;
        uint256[] memory epochIDs = new uint256[](1);
        epochIDs[0] = epochID;
        address user;
        uint256 assets;

        for (uint256 i = 0; i < users.length; i++) {
            user = users[i];
            assets = getUserEpochRequestInAssets(user, wwl.epochID, wwl.latestEpochID, wwl.epochPrice, decimals);
            if (assets == 0) continue;
            // accounting
            $.userEpochRequests[user][epochID] = 0;
            wwl.totalAssetsSum += assets;
            IERC20(asset).safeTransfer(user, assets);

            emit IConcreteAsyncVaultImpl.RequestClaimed(user, assets, epochIDs);
        }

        require(wwl.totalAssetsSum > 0, IConcreteAsyncVaultImpl.NoClaimableRequest());

        // Update accounting
        $.pastEpochsUnclaimedAssets -= wwl.totalAssetsSum;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Lifecycle Interaction Functions
    // ─────────────────────────────────────────────────────────────────────────────

    function moveRequestToNextEpoch(address user, uint256 fromEpochID, uint256 amount) public {
        ConcreteAsyncVaultImplStorageLib.ConcreteAsyncVaultImplStorage storage $ =
            ConcreteAsyncVaultImplStorageLib.fetch();

        uint256 latestEpoch = $.latestEpochID;
        uint256 closedEpochID = latestEpoch - 1; // closed epoch is the previous epoch
        require(fromEpochID == closedEpochID, IConcreteAsyncVaultImpl.EpochMismatch(fromEpochID, closedEpochID));
        require(!_epochProcessed(closedEpochID), IConcreteAsyncVaultImpl.EpochAlreadyProcessed(closedEpochID));

        uint256 userBalance = $.userEpochRequests[user][closedEpochID];
        require(userBalance > 0, IConcreteAsyncVaultImpl.NoRequestingShares());

        uint256 sharesToMove = amount == type(uint256).max ? userBalance : amount;
        require(sharesToMove > 0, IConcreteAsyncVaultImpl.ZeroShares());
        require(sharesToMove <= userBalance, IConcreteStandardVaultImpl.InsufficientShares());

        $.userEpochRequests[user][closedEpochID] -= sharesToMove;
        $.userEpochRequests[user][latestEpoch] += sharesToMove;
        $.totalRequestedSharesPerEpoch[closedEpochID] -= sharesToMove;
        $.totalRequestedSharesPerEpoch[latestEpoch] += sharesToMove;

        emit IConcreteAsyncVaultImpl.RequestMovedToNextEpoch(user, sharesToMove, closedEpochID, latestEpoch);
    }

    function cancelRequest(address user, uint256 epochID) public {
        ConcreteAsyncVaultImplStorageLib.ConcreteAsyncVaultImplStorage storage $ =
            ConcreteAsyncVaultImplStorageLib.fetch();

        require(_epochIsOpen(epochID), IConcreteAsyncVaultImpl.EpochAlreadyClosed(epochID));

        uint256 requestingShares = $.userEpochRequests[user][epochID];
        require(requestingShares > 0, IConcreteAsyncVaultImpl.NoRequestingShares());

        // Clear the request
        $.userEpochRequests[user][epochID] = 0;
        $.totalRequestedSharesPerEpoch[epochID] -= requestingShares;

        // Return shares to user
        ERC20_OZ_5_2_0_Lib._transfer(address(this), user, requestingShares);

        emit IConcreteAsyncVaultImpl.RequestCancelled(user, requestingShares, epochID);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // View Functions
    // ─────────────────────────────────────────────────────────────────────────────

    function getUserEpochRequestInAssets(
        address user,
        uint256 epochID,
        uint256 latestEpochID,
        uint256 epochPrice,
        uint8 decimals
    ) internal view returns (uint256 assets) {
        // Check if epoch is processed (has a price set)
        if (epochPrice == 0 || epochID >= latestEpochID) return 0;

        uint256 shares = ConcreteAsyncVaultImplStorageLib.fetch().userEpochRequests[user][epochID];
        if (shares == 0) return 0;

        assets = shares.mulDiv(epochPrice, 10 ** decimals, Math.Rounding.Floor);
    }

    function getEpochState(uint256 epochID) public view returns (IConcreteAsyncVaultImpl.EpochState) {
        uint256 latestEpochID = ConcreteAsyncVaultImplStorageLib.fetch().latestEpochID;
        if (epochID > latestEpochID) return IConcreteAsyncVaultImpl.EpochState.Inactive;
        if (epochID == latestEpochID) return IConcreteAsyncVaultImpl.EpochState.Active;
        bool isProcessed = _epochProcessed(epochID);
        if (isProcessed) return IConcreteAsyncVaultImpl.EpochState.Processed;
        return IConcreteAsyncVaultImpl.EpochState.Processing;
    }

    function totalRequestedSharesForCurrentEpochs()
        public
        view
        returns (uint256 activeShares, uint256 processingShares, uint256 processedShares)
    {
        ConcreteAsyncVaultImplStorageLib.ConcreteAsyncVaultImplStorage storage $ =
            ConcreteAsyncVaultImplStorageLib.fetch();
        uint256 activeEpochID = $.latestEpochID;
        activeShares = $.totalRequestedSharesPerEpoch[activeEpochID];
        uint256 previousEpochID = activeEpochID - 1;
        uint256 sharesInPreviousEpoch = $.totalRequestedSharesPerEpoch[previousEpochID];
        if (_epochProcessed(previousEpochID)) {
            processedShares = sharesInPreviousEpoch;
        } else {
            processingShares = sharesInPreviousEpoch;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Auxiliary Functions
    // ─────────────────────────────────────────────────────────────────────────────

    function _epochProcessed(uint256 epochID) private view returns (bool isProcessed) {
        (, isProcessed) = _epochPriceAndProcessedFlag(epochID);
    }

    function _epochPriceAndProcessedFlag(uint256 epochID) private view returns (uint256 epochPrice, bool isProcessed) {
        uint256 epochPricePlusOne = ConcreteAsyncVaultImplStorageLib.fetch().epochPricePerSharePlusOne[epochID];
        isProcessed = epochPricePlusOne > 0;
        epochPrice = isProcessed ? epochPricePlusOne - 1 : 0;
    }

    function _epochIsOpen(uint256 epochID) private view returns (bool) {
        return epochID >= ConcreteAsyncVaultImplStorageLib.fetch().latestEpochID;
    }

    function _requireStrategyRegisteredAndActive(address strategy) private view {
        SVLib.ConcreteStandardVaultImplStorage storage $std = SVLib.fetch();
        require(EnumerableSet.contains($std.strategies, strategy), IConcreteStandardVaultImpl.StrategyDoesNotExist());
        require(
            $std.strategyData[strategy].status == IConcreteStandardVaultImpl.StrategyStatus.Active,
            IConcreteStandardVaultImpl.StrategyIsHalted()
        );
    }

    function claimPriorityWithdrawal(
        address asset,
        address user,
        uint256 sharesToFulfil,
        uint256 unwindCost,
        uint256 lockedAssets,
        address strategy
    ) public {
        ConcreteAsyncVaultImplStorageLib.ConcreteAsyncVaultImplStorage storage $ =
            ConcreteAsyncVaultImplStorageLib.fetch();
        CachedVaultStateLib.ConcreteCachedVaultStateStorage storage $cached = CachedVaultStateLib.fetch();

        require(sharesToFulfil > 0, IConcreteAsyncVaultImpl.ZeroShares());
        require(user != address(0), IConcreteAsyncVaultImpl.ZeroAddress());
        require(
            sharesToFulfil <= $.userEpochRequests[user][$.latestEpochID],
            IConcreteStandardVaultImpl.InsufficientShares()
        );
        require(strategy != address(0), IConcreteAsyncVaultImpl.ZeroAddress());
        _requireStrategyRegisteredAndActive(strategy);

        // Calculate gross assets based on shares to fulfil and current exchange rate
        uint256 grossAssets = ConversionLib.calcConvertToAssets(
            sharesToFulfil, ERC20_OZ_5_2_0_Lib.totalSupply(), $cached.cachedTotalAssets, Math.Rounding.Floor, true
        );

        // Calculate maximum allowed unwind cost based on unwind cost cap
        uint256 maxUnwindCost = grossAssets.mulDiv($.unwindCostCapBP, BASIS_POINTS, Math.Rounding.Floor);
        require(unwindCost <= maxUnwindCost && unwindCost < grossAssets, IConcreteAsyncVaultImpl.UnwindCostTooHigh());

        uint256 netAssets = grossAssets - unwindCost;

        uint256 floatingFunds = IERC20(asset).balanceOf(address(this));
        uint256 availableAssets = floatingFunds > lockedAssets ? floatingFunds - lockedAssets : 0;
        require(availableAssets >= netAssets, IConcreteStandardVaultImpl.InsufficientBalance());

        // Update storage
        $.userEpochRequests[user][$.latestEpochID] -= sharesToFulfil;
        $.totalRequestedSharesPerEpoch[$.latestEpochID] -= sharesToFulfil;

        // Update cached total assets (subtract net assets sent out to user)
        $cached.cachedTotalAssets -= netAssets;

        // Burn the shares that are being fulfilled
        ERC20_OZ_5_2_0_Lib._burn(address(this), sharesToFulfil);

        // Adjust the strategy's reported allocation so the next yield accrual
        // reconciles the unwind cost without double-counting.
        // This also bumps the strategy's accounting nonce to prevent race
        // conditions with other accounting sources (e.g. operator adjustments).
        if (unwindCost > 0) {
            IAsyncAccounting(strategy).adjustTotalAssets(-int256(unwindCost));
        }

        // Transfer net assets to user
        IERC20(asset).safeTransfer(user, netAssets);

        emit IConcreteAsyncVaultImpl.PriorityWithdrawalClaimed(
            user, sharesToFulfil, grossAssets, unwindCost, netAssets, $.latestEpochID
        );
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/ERC20Lib.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import {ERC20Upgradeable} from "@openzeppelin-upgradeable/token/ERC20/ERC20Upgradeable.sol";
import {IERC20Errors} from "@openzeppelin-contracts/interfaces/draft-IERC6093.sol";
import {IERC20} from "@openzeppelin-contracts/token/ERC20/IERC20.sol";

library ERC20_OZ_5_2_0_Lib {
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
            revert IERC20Errors.ERC20InvalidSender(address(0));
        }
        if (to == address(0)) {
            revert IERC20Errors.ERC20InvalidReceiver(address(0));
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
    function _update(address from, address to, uint256 value) internal {
        ERC20Upgradeable.ERC20Storage storage $ = _getERC20Storage();
        if (from == address(0)) {
            // Overflow check required: The rest of the code assumes that totalSupply never overflows
            $._totalSupply += value;
        } else {
            uint256 fromBalance = $._balances[from];
            if (fromBalance < value) {
                revert IERC20Errors.ERC20InsufficientBalance(from, fromBalance, value);
            }
            unchecked {
                // Overflow not possible: value <= fromBalance <= totalSupply.
                $._balances[from] = fromBalance - value;
            }
        }

        if (to == address(0)) {
            unchecked {
                // Overflow not possible: value <= totalSupply or value <= fromBalance <= totalSupply.
                $._totalSupply -= value;
            }
        } else {
            unchecked {
                // Overflow not possible: balance + value is at most totalSupply, which we know fits into a uint256.
                $._balances[to] += value;
            }
        }

        emit IERC20.Transfer(from, to, value);
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
            revert IERC20Errors.ERC20InvalidReceiver(address(0));
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
            revert IERC20Errors.ERC20InvalidSender(address(0));
        }
        _update(account, address(0), value);
    }

    /**
     * @dev Sets `value` as the allowance of `spender` over the `owner` s tokens.
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
    function _approve(address owner, address spender, uint256 value, bool emitEvent) internal {
        ERC20Upgradeable.ERC20Storage storage $ = _getERC20Storage();
        if (owner == address(0)) {
            revert IERC20Errors.ERC20InvalidApprover(address(0));
        }
        if (spender == address(0)) {
            revert IERC20Errors.ERC20InvalidSpender(address(0));
        }
        $._allowances[owner][spender] = value;
        if (emitEvent) {
            emit IERC20.Approval(owner, spender, value);
        }
    }

    /**
     * @dev Updates `owner` s allowance for `spender` based on spent `value`.
     *
     * Does not update the allowance value in case of infinite allowance.
     * Revert if not enough allowance is available.
     *
     * Does not emit an {Approval} event.
     */
    function _spendAllowance(address owner, address spender, uint256 value) internal {
        uint256 currentAllowance = allowance(owner, spender);
        if (currentAllowance < type(uint256).max) {
            if (currentAllowance < value) {
                revert IERC20Errors.ERC20InsufficientAllowance(spender, currentAllowance, value);
            }
            unchecked {
                _approve(owner, spender, currentAllowance - value, false);
            }
        }
    }

    /**
     * @dev See {IERC20-allowance}.
     */
    function allowance(address owner, address spender) public view returns (uint256) {
        ERC20Upgradeable.ERC20Storage storage $ = _getERC20Storage();
        return $._allowances[owner][spender];
    }

    /**
     * @dev See {IERC20-totalSupply}.
     */
    function totalSupply() public view returns (uint256) {
        return _getERC20Storage()._totalSupply;
    }

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.ERC20")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant ERC20StorageLocation = 0x52c63247e1f47db19d5ce0460030c497f067ca4cebf71ba98eeadabe20bace00;

    function _getERC20Storage() private pure returns (ERC20Upgradeable.ERC20Storage storage $) {
        assembly {
            $.slot := ERC20StorageLocation
        }
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/lib/LegacyAdminRolesLib.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title LegacyAdminRolesLib
 * @dev Library containing legacy admin role definitions that were replaced by ROLE_ADMIN
 */
library LegacyAdminRolesLib {
    /// @dev Legacy admin roles (deprecated, replaced by ROLE_ADMIN)
    bytes32 public constant VAULT_MANAGER_ADMIN = keccak256("VAULT_MANAGER_ADMIN");
    bytes32 public constant HOOK_MANAGER_ADMIN = keccak256("HOOK_MANAGER_ADMIN");
    bytes32 public constant STRATEGY_MANAGER_ADMIN = keccak256("STRATEGY_MANAGER_ADMIN");
    bytes32 public constant ALLOCATOR_ADMIN = keccak256("ALLOCATOR_ADMIN");
    bytes32 public constant WITHDRAWAL_MANAGER_ADMIN = keccak256("WITHDRAWAL_MANAGER_ADMIN");
    bytes32 public constant PAUSER_ADMIN = keccak256("PAUSER_ADMIN");

    /**
     * @dev Returns all legacy admin roles as an array
     * @return roles Array of legacy admin role bytes32 values
     */
    function getAllLegacyAdminRoles() public pure returns (bytes32[] memory roles) {
        roles = new bytes32[](6);
        roles[0] = VAULT_MANAGER_ADMIN;
        roles[1] = HOOK_MANAGER_ADMIN;
        roles[2] = STRATEGY_MANAGER_ADMIN;
        roles[3] = ALLOCATOR_ADMIN;
        roles[4] = WITHDRAWAL_MANAGER_ADMIN;
        roles[5] = PAUSER_ADMIN;
    }
}



// ===== FILE: node_modules/_openzeppelin/contracts/access/IAccessControl.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (access/IAccessControl.sol)

pragma solidity ^0.8.20;

/**
 * @dev External interface of AccessControl declared to support ERC-165 detection.
 */
interface IAccessControl {
    /**
     * @dev The `account` is missing a role.
     */
    error AccessControlUnauthorizedAccount(address account, bytes32 neededRole);

    /**
     * @dev The caller of a function is not the expected one.
     *
     * NOTE: Don't confuse with {AccessControlUnauthorizedAccount}.
     */
    error AccessControlBadConfirmation();

    /**
     * @dev Emitted when `newAdminRole` is set as ``role``'s admin role, replacing `previousAdminRole`
     *
     * `DEFAULT_ADMIN_ROLE` is the starting admin for all roles, despite
     * {RoleAdminChanged} not being emitted signaling this.
     */
    event RoleAdminChanged(bytes32 indexed role, bytes32 indexed previousAdminRole, bytes32 indexed newAdminRole);

    /**
     * @dev Emitted when `account` is granted `role`.
     *
     * `sender` is the account that originated the contract call. This account bears the admin role (for the granted role).
     * Expected in cases where the role was granted using the internal {AccessControl-_grantRole}.
     */
    event RoleGranted(bytes32 indexed role, address indexed account, address indexed sender);

    /**
     * @dev Emitted when `account` is revoked `role`.
     *
     * `sender` is the account that originated the contract call:
     *   - if using `revokeRole`, it is the admin role bearer
     *   - if using `renounceRole`, it is the role bearer (i.e. `account`)
     */
    event RoleRevoked(bytes32 indexed role, address indexed account, address indexed sender);

    /**
     * @dev Returns `true` if `account` has been granted `role`.
     */
    function hasRole(bytes32 role, address account) external view returns (bool);

    /**
     * @dev Returns the admin role that controls `role`. See {grantRole} and
     * {revokeRole}.
     *
     * To change a role's admin, use {AccessControl-_setRoleAdmin}.
     */
    function getRoleAdmin(bytes32 role) external view returns (bytes32);

    /**
     * @dev Grants `role` to `account`.
     *
     * If `account` had not been already granted `role`, emits a {RoleGranted}
     * event.
     *
     * Requirements:
     *
     * - the caller must have ``role``'s admin role.
     */
    function grantRole(bytes32 role, address account) external;

    /**
     * @dev Revokes `role` from `account`.
     *
     * If `account` had been granted `role`, emits a {RoleRevoked} event.
     *
     * Requirements:
     *
     * - the caller must have ``role``'s admin role.
     */
    function revokeRole(bytes32 role, address account) external;

    /**
     * @dev Revokes `role` from the calling account.
     *
     * Roles are often managed via {grantRole} and {revokeRole}: this function's
     * purpose is to provide a mechanism for accounts to lose their privileges
     * if they are compromised (such as when a trusted device is misplaced).
     *
     * If the calling account had been granted `role`, emits a {RoleRevoked}
     * event.
     *
     * Requirements:
     *
     * - the caller must be `callerConfirmation`.
     */
    function renounceRole(bytes32 role, address callerConfirmation) external;
}


// ===== FILE: node_modules/_openzeppelin/contracts-upgradeable/utils/introspection/ERC165Upgradeable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/introspection/ERC165.sol)

pragma solidity ^0.8.20;

import {IERC165} from "@openzeppelin/contracts/utils/introspection/IERC165.sol";
import {Initializable} from "../../proxy/utils/Initializable.sol";

/**
 * @dev Implementation of the {IERC165} interface.
 *
 * Contracts that want to implement ERC-165 should inherit from this contract and override {supportsInterface} to check
 * for the additional interface id that will be supported. For example:
 *
 * ```solidity
 * function supportsInterface(bytes4 interfaceId) public view virtual override returns (bool) {
 *     return interfaceId == type(MyInterface).interfaceId || super.supportsInterface(interfaceId);
 * }
 * ```
 */
abstract contract ERC165Upgradeable is Initializable, IERC165 {
    function __ERC165_init() internal onlyInitializing {
    }

    function __ERC165_init_unchained() internal onlyInitializing {
    }
    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId) public view virtual returns (bool) {
        return interfaceId == type(IERC165).interfaceId;
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts/interfaces/IERC1363.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (interfaces/IERC1363.sol)

pragma solidity ^0.8.20;

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


// ===== FILE: node_modules/_openzeppelin/contracts/utils/Panic.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/Panic.sol)

pragma solidity ^0.8.20;

/**
 * @dev Helper library for emitting standardized panic codes.
 *
 * ```solidity
 * contract Example {
 *      using Panic for uint256;
 *
 *      // Use any of the declared internal constants
 *      function foo() { Panic.GENERIC.panic(); }
 *
 *      // Alternatively
 *      function foo() { Panic.panic(Panic.GENERIC); }
 * }
 * ```
 *
 * Follows the list from https://github.com/ethereum/solidity/blob/v0.8.24/libsolutil/ErrorCodes.h[libsolutil].
 *
 * _Available since v5.1._
 */
// slither-disable-next-line unused-state
library Panic {
    /// @dev generic / unspecified error
    uint256 internal constant GENERIC = 0x00;
    /// @dev used by the assert() builtin
    uint256 internal constant ASSERT = 0x01;
    /// @dev arithmetic underflow or overflow
    uint256 internal constant UNDER_OVERFLOW = 0x11;
    /// @dev division or modulo by zero
    uint256 internal constant DIVISION_BY_ZERO = 0x12;
    /// @dev enum conversion error
    uint256 internal constant ENUM_CONVERSION_ERROR = 0x21;
    /// @dev invalid encoding in storage
    uint256 internal constant STORAGE_ENCODING_ERROR = 0x22;
    /// @dev empty array pop
    uint256 internal constant EMPTY_ARRAY_POP = 0x31;
    /// @dev array out of bounds access
    uint256 internal constant ARRAY_OUT_OF_BOUNDS = 0x32;
    /// @dev resource error (too large allocation or too large array)
    uint256 internal constant RESOURCE_ERROR = 0x41;
    /// @dev calling invalid internal function
    uint256 internal constant INVALID_INTERNAL_FUNCTION = 0x51;

    /// @dev Reverts with a panic code. Recommended to use with
    /// the internal constants with predefined codes.
    function panic(uint256 code) internal pure {
        assembly ("memory-safe") {
            mstore(0x00, 0x4e487b71)
            mstore(0x20, code)
            revert(0x1c, 0x24)
        }
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts/utils/TransientSlot.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/TransientSlot.sol)
// This file was procedurally generated from scripts/generate/templates/TransientSlot.js.

pragma solidity ^0.8.24;

/**
 * @dev Library for reading and writing value-types to specific transient storage slots.
 *
 * Transient slots are often used to store temporary values that are removed after the current transaction.
 * This library helps with reading and writing to such slots without the need for inline assembly.
 *
 *  * Example reading and writing values using transient storage:
 * ```solidity
 * contract Lock {
 *     using TransientSlot for *;
 *
 *     // Define the slot. Alternatively, use the SlotDerivation library to derive the slot.
 *     bytes32 internal constant _LOCK_SLOT = 0xf4678858b2b588224636b8522b729e7722d32fc491da849ed75b3fdf3c84f542;
 *
 *     modifier locked() {
 *         require(!_LOCK_SLOT.asBoolean().tload());
 *
 *         _LOCK_SLOT.asBoolean().tstore(true);
 *         _;
 *         _LOCK_SLOT.asBoolean().tstore(false);
 *     }
 * }
 * ```
 *
 * TIP: Consider using this library along with {SlotDerivation}.
 */
library TransientSlot {
    /**
     * @dev UDVT that represent a slot holding a address.
     */
    type AddressSlot is bytes32;

    /**
     * @dev Cast an arbitrary slot to a AddressSlot.
     */
    function asAddress(bytes32 slot) internal pure returns (AddressSlot) {
        return AddressSlot.wrap(slot);
    }

    /**
     * @dev UDVT that represent a slot holding a bool.
     */
    type BooleanSlot is bytes32;

    /**
     * @dev Cast an arbitrary slot to a BooleanSlot.
     */
    function asBoolean(bytes32 slot) internal pure returns (BooleanSlot) {
        return BooleanSlot.wrap(slot);
    }

    /**
     * @dev UDVT that represent a slot holding a bytes32.
     */
    type Bytes32Slot is bytes32;

    /**
     * @dev Cast an arbitrary slot to a Bytes32Slot.
     */
    function asBytes32(bytes32 slot) internal pure returns (Bytes32Slot) {
        return Bytes32Slot.wrap(slot);
    }

    /**
     * @dev UDVT that represent a slot holding a uint256.
     */
    type Uint256Slot is bytes32;

    /**
     * @dev Cast an arbitrary slot to a Uint256Slot.
     */
    function asUint256(bytes32 slot) internal pure returns (Uint256Slot) {
        return Uint256Slot.wrap(slot);
    }

    /**
     * @dev UDVT that represent a slot holding a int256.
     */
    type Int256Slot is bytes32;

    /**
     * @dev Cast an arbitrary slot to a Int256Slot.
     */
    function asInt256(bytes32 slot) internal pure returns (Int256Slot) {
        return Int256Slot.wrap(slot);
    }

    /**
     * @dev Load the value held at location `slot` in transient storage.
     */
    function tload(AddressSlot slot) internal view returns (address value) {
        assembly ("memory-safe") {
            value := tload(slot)
        }
    }

    /**
     * @dev Store `value` at location `slot` in transient storage.
     */
    function tstore(AddressSlot slot, address value) internal {
        assembly ("memory-safe") {
            tstore(slot, value)
        }
    }

    /**
     * @dev Load the value held at location `slot` in transient storage.
     */
    function tload(BooleanSlot slot) internal view returns (bool value) {
        assembly ("memory-safe") {
            value := tload(slot)
        }
    }

    /**
     * @dev Store `value` at location `slot` in transient storage.
     */
    function tstore(BooleanSlot slot, bool value) internal {
        assembly ("memory-safe") {
            tstore(slot, value)
        }
    }

    /**
     * @dev Load the value held at location `slot` in transient storage.
     */
    function tload(Bytes32Slot slot) internal view returns (bytes32 value) {
        assembly ("memory-safe") {
            value := tload(slot)
        }
    }

    /**
     * @dev Store `value` at location `slot` in transient storage.
     */
    function tstore(Bytes32Slot slot, bytes32 value) internal {
        assembly ("memory-safe") {
            tstore(slot, value)
        }
    }

    /**
     * @dev Load the value held at location `slot` in transient storage.
     */
    function tload(Uint256Slot slot) internal view returns (uint256 value) {
        assembly ("memory-safe") {
            value := tload(slot)
        }
    }

    /**
     * @dev Store `value` at location `slot` in transient storage.
     */
    function tstore(Uint256Slot slot, uint256 value) internal {
        assembly ("memory-safe") {
            tstore(slot, value)
        }
    }

    /**
     * @dev Load the value held at location `slot` in transient storage.
     */
    function tload(Int256Slot slot) internal view returns (int256 value) {
        assembly ("memory-safe") {
            value := tload(slot)
        }
    }

    /**
     * @dev Store `value` at location `slot` in transient storage.
     */
    function tstore(Int256Slot slot, int256 value) internal {
        assembly ("memory-safe") {
            tstore(slot, value)
        }
    }
}


// ===== FILE: node_modules/_concrete-xyz/earn-v2-core/src/interface/IAsyncAccounting.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

/**
 * @title IAsyncAccounting
 * @notice Interface for strategies that use asynchronous (off-chain) accounting.
 * @dev Strategies whose reported `totalAllocatedValue()` depends on an off-chain
 *      accounting update (e.g. a multisig-managed position) should implement this
 *      interface.  It exposes:
 *
 *      - Two overloads of `adjustTotalAssets`:
 *        • `adjustTotalAssets(int256 diff)` — vault-initiated, trusted, no validation.
 *        • `adjustTotalAssets(int256 diff, uint256 nonce)` — operator-initiated with
 *          nonce / threshold / cooldown validation.
 *
 *      - `unpauseAndAdjustTotalAssets` — admin emergency recovery.
 *
 *      - Configuration setters for the accounting validation parameters.
 *
 *      - Read-only getters for the current accounting configuration.
 */
interface IAsyncAccounting {
    // ─── Adjust functions ────────────────────────────────────────────────

    /**
     * @notice Vault-initiated accounting adjustment (e.g. recording an unwind cost).
     * @dev MUST only be callable by the bound vault.  Bypasses nonce / threshold /
     *      cooldown checks because the vault is a trusted caller.  MUST bump the
     *      accounting nonce to prevent race conditions with operator adjustments.
     * @param diff Signed delta to apply to the strategy's internal position value.
     */
    function adjustTotalAssets(int256 diff) external;

    /**
     * @notice Operator-initiated accounting adjustment with validation.
     * @dev Subject to nonce ordering, cooldown period, and max-change-threshold
     *      checks.  If validation fails the strategy SHOULD pause itself.
     * @param diff  Signed delta to apply.
     * @param nonce The expected next accounting nonce (prevents replay / ordering issues).
     */
    function adjustTotalAssets(int256 diff, uint256 nonce) external;

    /**
     * @notice Emergency admin function: unpause the strategy and apply an adjustment
     *         in a single transaction, bypassing threshold / cooldown checks.
     * @param diff Signed delta to apply after unpausing.
     */
    function unpauseAndAdjustTotalAssets(int256 diff) external;

    // ─── Configuration setters ───────────────────────────────────────────

    /**
     * @notice Set the maximum allowed single-adjustment change as a fraction of the
     *         current position, expressed in basis points (10 000 = 100 %).
     * @param maxAccountingChangeThreshold New threshold in basis points.
     */
    function setMaxAccountingChangeThreshold(uint64 maxAccountingChangeThreshold) external;

    /**
     * @notice Set the validity window for accounting data.  If no adjustment is
     *         made within this period the strategy's `totalAllocatedValue` will
     *         revert, signalling stale data.
     * @param accountingValidityPeriod New validity period in seconds.
     */
    function setAccountingValidityPeriod(uint64 accountingValidityPeriod) external;

    /**
     * @notice Set the minimum time that must elapse between two successive
     *         operator-initiated adjustments.
     * @param cooldownPeriod New cooldown period in seconds.
     */
    function setCooldownPeriod(uint64 cooldownPeriod) external;

    // ─── Getters ─────────────────────────────────────────────────────────

    /**
     * @notice Returns the next expected accounting nonce.
     */
    function getNextAccountingNonce() external view returns (uint256);

    /**
     * @notice Returns the timestamp of the most recent accounting adjustment.
     */
    function getLastUpdatedTimestamp() external view returns (uint64);

    /**
     * @notice Returns the current max-change threshold in basis points.
     */
    function getMaxAccountingChangeThreshold() external view returns (uint64);

    /**
     * @notice Returns the current accounting validity period in seconds.
     */
    function getAccountingValidityPeriod() external view returns (uint64);

    /**
     * @notice Returns the current cooldown period in seconds.
     */
    function getCooldownPeriod() external view returns (uint64);
}


// ===== FILE: node_modules/_openzeppelin/contracts/utils/introspection/IERC165.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/introspection/IERC165.sol)

pragma solidity ^0.8.20;

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


// ===== FILE: node_modules/_openzeppelin/contracts/interfaces/IERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (interfaces/IERC20.sol)

pragma solidity ^0.8.20;

import {IERC20} from "../token/ERC20/IERC20.sol";


// ===== FILE: node_modules/_openzeppelin/contracts/interfaces/IERC165.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (interfaces/IERC165.sol)

pragma solidity ^0.8.20;

import {IERC165} from "../utils/introspection/IERC165.sol";
