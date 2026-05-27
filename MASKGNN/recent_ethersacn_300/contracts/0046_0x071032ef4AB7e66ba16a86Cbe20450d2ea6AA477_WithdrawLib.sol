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
