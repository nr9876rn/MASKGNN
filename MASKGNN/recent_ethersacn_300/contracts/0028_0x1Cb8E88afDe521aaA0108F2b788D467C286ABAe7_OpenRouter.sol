// ===== FILE: src/OpenRouter.sol =====
// SPDX-License-Identifier: GPL-3.0-only
pragma solidity 0.8.34;

import {SafeTransferLib} from "solady/src/utils/SafeTransferLib.sol";

import {IERC20} from "./common/interfaces/IERC20.sol";
import {AccessControl} from "./common/utils/AccessControl.sol";
import {AllowanceHolderContext} from "./common/allowance/AllowanceHolderContext.sol";
import {ALLOWANCE_HOLDER} from "./common/interfaces/IAllowanceHolder.sol";
import {BytesSpliceLib} from "./common/lib/BytesSpliceLib.sol";
import {CurrencyLib} from "./common/lib/CurrencyLib.sol";
import {RescueFundsLib} from "./common/lib/RescueFundsLib.sol";
import {RESCUE_ROLE} from "./common/AccessRoles.sol";

/// @title OpenRouter
/// @notice Pull → optional fee → swap/bridge execution without backend signature verification.
///         Fund safety rests on AllowanceHolder's transient allowance scoping (operator + owner + token):
///         only the user whose address was passed to `AllowanceHolder.exec` can authorise a pull of
///         their own funds. The `_msgSender() == user` check in `_pullFromUser` enforces this.
contract OpenRouter is AccessControl, AllowanceHolderContext {
    using SafeTransferLib for address;

    // =========================================================================
    // Structs
    // =========================================================================

    struct InputData {
        address user;
        address inputToken;
        uint256 inputAmount;
    }

    struct FeeData {
        address receiver;
        uint256 amount;
    }

    struct SwapData {
        address target;
        address approvalSpender;
        address outputToken;
        uint256 value;
        uint256 minOutput;
        uint256 returnDataWordOffset;
    }

    struct BridgeData {
        address target;
        address approvalSpender;
        uint256 value;
    }

    enum CallType {
        CALL,
        STATICCALL,
        CALL_WITH_NATIVE
    }

    struct Action {
        /// @dev Packed call metadata. Decode with masks/shifts below; encode with
        ///      `callType | (storeResult ? 1 << 8 : 0) | (uint160(target) << 16)`.
        ///
        /// Bit layout (least significant bits first):
        ///   bits 255..160 : reserved (0)
        ///   bits 159..16  : target address (uint160, left-aligned in this field)
        ///   bit 8         : storeResult — when set, returndata is saved to `results[i]`
        ///                   even on success so later actions can splice from it
        ///   bits 7..3     : reserved (0)
        ///   bits 2..0     : CallType — CALL (0), STATICCALL (1), CALL_WITH_NATIVE (2)
        ///
        /// CALL_WITH_NATIVE: first 32 bytes of `data` are forwarded as `msg.value`;
        /// the remaining bytes are the call payload.
        uint256 actionInfo;
        /// @dev Calldata passed to the target. Splices from `splices[]` overwrite byte
        /// ranges in a mutable memory copy before the external call runs.
        bytes data;
        /// @dev Packed splice descriptors applied to `data` before the call.
        /// Each entry is one `uint256` with four uint64 fields (see layout below).
        /// Encode with `packSpliceInfo` in `scripts/e2e/utils/modularActionsBuilder/index.js`.
        ///
        /// Per-entry bit layout (least significant bits first):
        ///   bits 255..192 : length — number of bytes to copy (must be > 0)
        ///   bits 191..128 : dstOffset — byte offset into this action's `data` payload
        ///                   (skips the bytes-array length word; for CALL_WITH_NATIVE,
        ///                   offset 0 is the value word, offset 32 is payload start)
        ///   bits 127..64  : srcOffset — byte offset into `results[sourceActionIndex]`
        ///                   payload (same length-prefix convention)
        ///   bits 63..0    : sourceActionIndex — index of a prior action (< current index)
        ///
        /// Packing formula:
        ///   sourceActionIndex | (srcOffset << 64) | (dstOffset << 128) | (length << 192)
        ///
        /// The source action must have bit 8 set in `actionInfo` (storeResult); the JS
        /// builder sets this automatically when a splice references that action.
        uint256[] splices;
    }

    // =========================================================================
    // Flags (swap / swapAndBridge)
    // =========================================================================
    //
    // Instead of bool parameters, one uint256 packs independent switches without adding
    // ABI range checks or extra words for standalone bools.
    //
    // Bit layout (least significant bits); test with `(flags & MASK) != 0`:
    //   bits 255..32 : reserved (0)
    //   bits 31..16 : bridge amount word byte offset, uint16, used only when bit 3 is set
    //   bits 15..4  : reserved (0)
    //   bit 3     : BRIDGE_AMOUNT_POSITION_FLAG_BIT_MASK (0x08) — splice finalAmount into bridge calldata
    //   bit 2     : BRIDGE_VALUE_FLAG_BIT_MASK (0x04) — bridge msg.value: bridge.value alone vs finalAmount + bridge.value
    //   bit 1     : BALANCE_FLAG_BIT_MASK (0x02) — swap output: returndata vs balance delta
    //   bit 0     : POST_FEE_FLAG_BIT_MASK (0x01)   — swap fee: pre- vs post-swap
    //
    // Combined values for flags:
    //
    //   flags  binary (low byte)    postFee?   balance-of output?            bridge value?
    //   ─────  ──────────────────  ────────   ──────────────────             ─────────────
    //   0x00   00000000              no         returndata word               bridge.value
    //   0x01   00000001              yes        returndata word               bridge.value
    //   0x02   00000010              no         balance delta on outputToken  bridge.value
    //   0x03   00000011              yes        balance delta on outputToken  bridge.value
    //   0x04   00000100              no         returndata word               finalAmount + bridge.value
    //
    // POST_FEE_FLAG_BIT_MASK selects bit 0 — fee timing
    //   0000 — pre-swap fee: pull → deduct fee from input token → swap remainder
    //   0001 — post-swap fee: pull → swap full input → deduct fee from output token (after minOutput check on swap result)
    //
    // BALANCE_FLAG_BIT_MASK selects bit 1 — swap output sizing
    //   0000 — returnData as swap output: decode returned amount from call returndata at `swapData.returnDataWordOffset`
    //   0010 — balanceOf() delta as swap output: snapshot outputToken balance before call, measure (after − before) as output
    //
    // BRIDGE_VALUE_FLAG_BIT_MASK selects bit 2 — bridge native value source
    //   0000 — bridge.value as msg.value: forward `bridge.value` as msg.value
    //   0100 — finalAmount + bridge.value as msg.value: forward `finalAmount + bridge.value` as msg.value (bridge.value carries static addend, e.g. LZ nativeFee)
    //
    // BRIDGE_AMOUNT_POSITION_FLAG_BIT_MASK selects bit 3 — bridge calldata amount splicing.
    //   0000 — no bridge calldata modification
    //   1000 — bridge calldata modification: splice finalAmount at uint16(flags >> BRIDGE_AMOUNT_POSITION_SHIFT)
    //

    /// @dev Bit mask 0x01: post-swap fee path when `(flags & mask) != 0`; clear = pre-swap fee from input token.
    uint256 internal constant POST_FEE_FLAG_BIT_MASK = 0x01;

    /// @dev Bit mask 0x02: measure swap output by balance delta when `(flags & mask) != 0`; clear = returndata word.
    uint256 internal constant BALANCE_FLAG_BIT_MASK = 0x02;

    /// @dev Bit mask 0x04: `finalAmount + bridge.value` is forwarded as msg.value (bridge.value acts as a static addend, e.g. LZ nativeFee).
    uint256 internal constant BRIDGE_VALUE_FLAG_BIT_MASK = 0x04;

    /// @dev Bit mask 0x08: splice finalAmount into bridge calldata at the uint16 position packed in flags.
    uint256 internal constant BRIDGE_AMOUNT_POSITION_FLAG_BIT_MASK = 0x08;

    /// @dev Shift for the packed uint16 bridge amount position.
    uint256 internal constant BRIDGE_AMOUNT_POSITION_SHIFT = 16;

    /// @dev Mask for the packed uint16 bridge amount position after shifting.
    uint256 internal constant BRIDGE_AMOUNT_POSITION_MASK = 0xffff;

    // =========================================================================
    // Errors
    // =========================================================================

    error SwapOutputInsufficient();
    error InvalidExecution();
    error CallerNotSignedUser();
    error InsufficientMsgValue();
    error FutureSplice(uint256 actionIndex, uint256 sourceActionIndex);
    error SpliceOutOfBounds(uint256 actionIndex, uint256 spliceIndex);
    error CallFailed(uint256 actionIndex, bytes returndata);
    error MissingNativeValue(uint256 actionIndex);
    error ReturnDataOutOfBounds();

    // =========================================================================
    // Events
    // =========================================================================

    event RequestExecuted(bytes32 indexed quoteId);

    // =========================================================================
    // Constructor
    // =========================================================================

    /**
     * @notice Deploys the router and grants `RESCUE_ROLE` to `_owner`.
     * @param _owner Initial contract owner and rescue-role holder.
     */
    constructor(address _owner) AccessControl(_owner) {
        _grantRole(RESCUE_ROLE, _owner);
    }

    /// @notice Accepts native ETH forwarded with bridge/swap calls.
    receive() external payable {}

    // =========================================================================
    // External functions
    // =========================================================================

    /**
     * @notice Perform swap with optional pre/post fee.
     * @param quoteId Caller-defined correlation id logged in `RequestExecuted`.
     * @param flags Packed flags
     * @param input User, input token, and pull amount.
     * @dev For pre-fee / no-fee: the swap router must
     *      be instructed (via `swapCallData`) to send tokens directly to `receiver`; the contract never holds the output.
     *      For post-fee: tokens land at this contract, fee is deducted, net is forwarded to `receiver`.
     * @param fee Fee collection info: receiver and amount. Set `amount` to 0 to skip fee collection.
     * @param swapData Swap target, spender, output token, value, `minOutput`, and returndata offset.
     * @param swapCallData Calldata forwarded to `swapData.target`.
     * @param receiver Address that ultimately receives the swap output (net of any post-swap fee).
     * @return finalAmount Gross swap output sent to receiver after any post-swap fee
     * @dev `minOutput` is the minimum gross amount coming out of the swap (before any output-token fee). It is enforced immediately after `_execSwap`, then post-swap fee (if any) is collected.
     *      Pre-fee paths take the input-side fee before the swap; `minOutput` still guards the swap outcome.
     */
    function swap(
        bytes32 quoteId,
        uint256 flags,
        InputData calldata input,
        FeeData calldata fee,
        SwapData calldata swapData,
        bytes calldata swapCallData,
        address receiver
    ) external payable returns (uint256 finalAmount) {
        if (
            input.user == address(0) || input.inputToken == address(0) || swapData.target == address(0)
                || receiver == address(0)
        ) {
            revert InvalidExecution();
        }

        // Parse flags
        bool postFee = fee.amount != 0 && ((flags & POST_FEE_FLAG_BIT_MASK) != 0);
        bool useBalanceOf = ((flags & BALANCE_FLAG_BIT_MASK) != 0);

        {
            // Pull funds from user via AllowanceHolder
            _pullFromUser(input.inputToken, input.user, input.inputAmount);

            // Collect pre-swap fee
            uint256 swapInput = input.inputAmount;
            if (fee.amount != 0 && !postFee) {
                uint256 feeAmount = fee.amount;
                CurrencyLib.transfer(input.inputToken, fee.receiver, feeAmount);
                unchecked {
                    swapInput -= feeAmount;
                }
            }

            // Approve spender
            if (
                // check spender & token
                swapData.approvalSpender != address(0) && input.inputToken != CurrencyLib.NATIVE_TOKEN_ADDRESS && 
                    // check current allowance
                    swapInput > IERC20(input.inputToken).allowance(address(this), swapData.approvalSpender)
            ) {
                // approve max allowance
                SafeTransferLib.safeApproveWithRetry(input.inputToken, swapData.approvalSpender, type(uint256).max);
            }
        }

        /// @dev Pre-fee / no-fee: swap calldata encodes `receiver` as the output recipient; tokens never touch this contract.
        /// @dev Post-fee: swap output lands at this contract so the fee can be deducted before forwarding.
        address outputReceiver = postFee ? address(this) : receiver;

        // Execute swap
        finalAmount = _execSwap(swapData, swapCallData, useBalanceOf, outputReceiver);
        if (finalAmount < swapData.minOutput) revert SwapOutputInsufficient();

        if (postFee) {
            // Collect post-swap fee
            uint256 feeAmount = fee.amount;
            CurrencyLib.transfer(swapData.outputToken, fee.receiver, feeAmount);
            unchecked {
                finalAmount -= feeAmount;
            }

            // Transfer net output to receiver
            CurrencyLib.transfer(swapData.outputToken, receiver, finalAmount);
        }

        // Pre-fee / no-fee: tokens were sent directly to `receiver` by the swap router; nothing to transfer

        emit RequestExecuted(quoteId);
    }

    /**
     * @notice Perform swap and bridge with optional pre/post swap fee.
     * @param quoteId Caller-defined correlation id logged in `RequestExecuted`.
     * @param flags Packed flags
     * @param input User, input token, and pull amount.
     * @param fee Fee collection info: receiver and amount. Set `amount` to 0 to skip fee collection.
     * @param swapData Swap target, spender, output token, value, `minOutput`, and returndata offset.
     * @param swapCallData Calldata forwarded to `swapData.target`.
     * @param bridgeData Bridge target, approval spender, and static `msg.value` addend.
     * @param bridgeCallData Bridge calldata; optionally spliced with swap output per `flags`.
     * @dev Same `minOutput` rule as `swap`: validated on gross `_execSwap` output, then optional output fee applies.
     */
    function swapAndBridge(
        bytes32 quoteId,
        uint256 flags,
        InputData calldata input,
        FeeData calldata fee,
        SwapData calldata swapData,
        bytes calldata swapCallData,
        BridgeData calldata bridgeData,
        bytes calldata bridgeCallData
    ) external payable {
        if (
            bridgeData.target == address(0) || input.user == address(0) || input.inputToken == address(0)
                || swapData.target == address(0)
        ) {
            revert InvalidExecution();
        }

        // Execute swap before bridge
        uint256 finalAmount = _swapBeforeBridge(flags, input, fee, swapData, swapCallData);

        // Execute bridge
        _execBridge(swapData.outputToken, finalAmount, flags, bridgeData, bridgeCallData);

        emit RequestExecuted(quoteId);
    }

    /**
     * @notice Perform bridge with optional pre-bridge fee.
     * @param quoteId Caller-defined correlation id logged in `RequestExecuted`.
     * @param input User, input token, and pull amount.
     * @param fee Fee collection info: receiver and amount. Set `amount` to 0 to skip fee collection.
     * @param bridgeData Bridge target, approval spender, and `msg.value` for the bridge call.
     * @param bridgeCallData Calldata forwarded to `bridgeData.target` (amount must be baked in by the caller).
     * @dev Because no swap is involved, `finalAmount = inputAmount - feeAmount` is fully knowable by the caller before signing.
     *      The caller must therefore bake the correct amount directly into `bridgeCallData` and set `bridgeData.value` to the desired `msg.value` for the bridge call.
     *      No runtime calldata splicing is performed. The caller MUST route through `AllowanceHolder.exec` for ERC-20 inputs so that `_msgSender()` resolves to `input.user`.
     */
    function bridge(
        bytes32 quoteId,
        InputData calldata input,
        FeeData calldata fee,
        BridgeData calldata bridgeData,
        bytes calldata bridgeCallData
    ) external payable {
        if (bridgeData.target == address(0) || input.user == address(0) || input.inputToken == address(0)) {
            revert InvalidExecution();
        }

        // Pull funds from user via AllowanceHolder
        _pullFromUser(input.inputToken, input.user, input.inputAmount);

        // Collect pre-bridge fee
        uint256 feeAmount = fee.amount;
        if (feeAmount != 0) {
            CurrencyLib.transfer(input.inputToken, fee.receiver, feeAmount);
        }

        uint256 netAmount;
        unchecked {
            netAmount = input.inputAmount - feeAmount;
        }

        // Approve bridge spender
        if (
            // check spender && token
            bridgeData.approvalSpender != address(0) && input.inputToken != CurrencyLib.NATIVE_TOKEN_ADDRESS && 
                // check current allowance
                netAmount > IERC20(input.inputToken).allowance(address(this), bridgeData.approvalSpender)
        ) {
            // approve max allowance
            SafeTransferLib.safeApproveWithRetry(input.inputToken, bridgeData.approvalSpender, type(uint256).max);
        }

        // Execute bridge
        _execCallCalldata(bridgeData.target, bridgeData.value, bridgeCallData, false);

        emit RequestExecuted(quoteId);
    }

    /**
     * @notice Runs a sequence of generic actions with optional returndata splicing between steps.
     * @param quoteId Caller-defined correlation id logged in `RequestExecuted`.
     * @param actions Ordered actions; each may splice bytes from a prior action's returndata into its calldata.
     */
    function performActions(bytes32 quoteId, Action[] calldata actions) external payable {
        _performActions(actions);

        emit RequestExecuted(quoteId);
    }

    // =========================================================================
    // Internal functions
    // =========================================================================

    // -------------------------------------
    //   swapAndBridge internal functions
    // -------------------------------------

    /**
     * @dev Pull, optional pre/post swap fee, and swap for `swapAndBridge`. Swap output always remains at `address(this)` for bridging.
     * @param flags Fee timing and swap output measurement flags (same as `swap`).
     * @param input User, input token, and pull amount.
     * @param fee Fee receiver and amount; `amount == 0` skips fee collection.
     * @param swapData Swap target, spender, output token, value, `minOutput`, and returndata offset.
     * @param swapCallData Calldata forwarded to `swapData.target`.
     * @return finalAmount Swap output net of any post-swap fee, ready for `_execBridge`.
     */
    function _swapBeforeBridge(
        uint256 flags,
        InputData calldata input,
        FeeData calldata fee,
        SwapData calldata swapData,
        bytes calldata swapCallData
    ) internal returns (uint256 finalAmount) {
        // Pull funds from user via AllowanceHolder
        _pullFromUser(input.inputToken, input.user, input.inputAmount);

        bool postFee;
        {
            // Collect pre-swap fee
            uint256 feeAmount = fee.amount;
            postFee = feeAmount != 0 && ((flags & POST_FEE_FLAG_BIT_MASK) != 0);
            uint256 swapInput = input.inputAmount;

            if (feeAmount != 0 && !postFee) {
                CurrencyLib.transfer(input.inputToken, fee.receiver, feeAmount);
                unchecked {
                    swapInput -= feeAmount;
                }
            }

            // Approve swap spender
            if (
                // check spender & token
                swapData.approvalSpender != address(0) && input.inputToken != CurrencyLib.NATIVE_TOKEN_ADDRESS && 
                    // check current allowance
                    swapInput > IERC20(input.inputToken).allowance(address(this), swapData.approvalSpender)
            ) {
                // approve max allowance
                SafeTransferLib.safeApproveWithRetry(input.inputToken, swapData.approvalSpender, type(uint256).max);
            }
        }

        // Execute swap
        /// @dev Swap output always lands at this contract regardless of fee timing — tokens must be here for bridging.
        bool useBalanceOf = ((flags & BALANCE_FLAG_BIT_MASK) != 0);
        finalAmount = _execSwap(swapData, swapCallData, useBalanceOf, address(this));
        if (finalAmount < swapData.minOutput) revert SwapOutputInsufficient();

        // Collect post-swap fee
        if (postFee) {
            uint256 feeAmount = fee.amount;
            CurrencyLib.transfer(swapData.outputToken, fee.receiver, feeAmount);
            unchecked {
                finalAmount -= feeAmount;
            }
        }
    }

    /**
     * @dev Splice `amount` into bridge calldata when flagged, approve the bridge spender, and call the bridge target.
     * @param token ERC-20 bridged (or native sentinel); used for approval only.
     * @param amount Post-swap token amount spliced into calldata and/or forwarded as `msg.value`.
     * @param flags Bridge splice position, `msg.value` composition, and related bit flags.
     * @param bridgeData Bridge target, approval spender, and static `msg.value` addend.
     * @param bridgeCallData Base bridge calldata; copied to memory when splicing is required.
     */
    function _execBridge(
        address token,
        uint256 amount,
        uint256 flags,
        BridgeData calldata bridgeData,
        bytes calldata bridgeCallData
    ) internal {
        bytes memory _bridgeCallData = bridgeCallData;

        // Modify bridge calldata if splicing is required
        if (flags & BRIDGE_AMOUNT_POSITION_FLAG_BIT_MASK != 0) {
            uint256 position = flags >> BRIDGE_AMOUNT_POSITION_SHIFT & BRIDGE_AMOUNT_POSITION_MASK;
            BytesSpliceLib.spliceWord({data: _bridgeCallData, position: position, word: amount});
        }

        // Approve bridge spender
        if (
            // check spender & token
            bridgeData.approvalSpender != address(0) && token != CurrencyLib.NATIVE_TOKEN_ADDRESS && 
                // check current allowance
                amount > IERC20(token).allowance(address(this), bridgeData.approvalSpender)
        ) {
            // approve max allowance
            SafeTransferLib.safeApproveWithRetry(token, bridgeData.approvalSpender, type(uint256).max);
        }

        // Parse and set bridge value flag
        uint256 bridgeValue = ((flags & BRIDGE_VALUE_FLAG_BIT_MASK) != 0) ? amount + bridgeData.value : bridgeData.value;

        // Execute bridge call
        _execCall(bridgeData.target, bridgeValue, _bridgeCallData);
    }

    // --------------------------------------
    //   performActions internal functions
    // --------------------------------------

    /**
     * @dev Executes `actions` in order, applying returndata splices before each call.
     * @dev See `Action` for `actionInfo` and `splices[]` bit layouts.
     * @param actions Ordered list of actions to run.
     */
    function _performActions(Action[] calldata actions) internal {
        uint256 actionsLength = actions.length;
        bytes[] memory results = new bytes[](actionsLength);

        for (uint256 i; i < actionsLength;) {
            Action calldata action = actions[i];
            bytes memory callData = action.data;

            // Patch callData with slices of prior action returndata.
            uint256 splicesLength = action.splices.length;
            for (uint256 j; j < splicesLength;) {
                uint256 spliceInfo = action.splices[j];
                uint256 sourceActionIndex = uint64(spliceInfo); // first 64 bits: index of the prior action to read returndata from.
                if (sourceActionIndex >= i) revert FutureSplice(i, sourceActionIndex);

                uint256 srcOffset = uint64(spliceInfo >> 64); // Next 64 bits: byte offset into source returndata
                uint256 dstOffset = uint64(spliceInfo >> 128); // Next 64 bits: byte offset into next action's data
                uint256 length = spliceInfo >> 192; // Top 64 bits: number of bytes to copy

                // Fetch source action returndata
                bytes memory source = results[sourceActionIndex];
                if (srcOffset + length > source.length || dstOffset + length > callData.length) {
                    revert SpliceOutOfBounds(i, j);
                }

                assembly ("memory-safe") {
                    // copy `length` bytes from `source returndata starting from `srcOffset` to `callData` starting from `dstOffset`
                    mcopy(add(add(callData, 0x20), dstOffset), add(add(source, 0x20), srcOffset), length)
                }

                unchecked {
                    ++j;
                }
            }

            // Parse actionInfo
            bool success;
            uint256 actionInfo = action.actionInfo;
            bool storeResult = (actionInfo & 0xff00) != 0; // Bit 8: persist returndata if set
            uint256 callType = actionInfo & 0xff; // Bits 0–7: specify CallType
            address target = address(uint160(actionInfo >> 16)); // Bits 16+: target address

            if (callType == uint256(CallType.STATICCALL)) {
                assembly ("memory-safe") {
                    // staticcall without copying return data by default
                    success := staticcall(gas(), target, add(callData, 0x20), mload(callData), 0, 0)
                }
            } else if (callType == uint256(CallType.CALL_WITH_NATIVE)) {
                if (callData.length < 32) revert MissingNativeValue(i);
                uint256 callValue;
                uint256 payloadLength = callData.length - 32;
                assembly ("memory-safe") {
                    // regular call with value forwarded without copying return data by default
                    callValue := mload(add(callData, 0x20)) // CALL_WITH_NATIVE prepends a 32-byte wei amount before the actual calldata payload.
                    success := call(gas(), target, callValue, add(callData, 0x40), payloadLength, 0, 0) // skips first two bytes to reach actuall calldata
                }
            } else {
                assembly ("memory-safe") {
                    // regular call with zero value forwarded without copying return data by default
                    success := call(gas(), target, 0, add(callData, 0x20), mload(callData), 0, 0)
                }
            }

            // Capture returndata on failure (for revert reason) or when explicitly requested.
            if (!success || storeResult) {
                bytes memory ret;
                assembly ("memory-safe") {
                    // prep return / revert data
                    let returnDataSize := returndatasize()
                    ret := mload(0x40)
                    mstore(ret, returnDataSize) // write length prefix on free-mem pointer
                    returndatacopy(add(ret, 0x20), 0, returnDataSize) // copy returndata after length
                    mstore(0x40, and(add(add(add(ret, 0x20), returnDataSize), 0x1f), not(0x1f))) // Advance free pointer to next 32-byte boundary: (ret + 0x20 + size + 31) and clear last 5 bits with not(0x1f)
                }
                // if any call was failed, revert with the returndata
                if (!success) revert CallFailed(i, ret);
                
                // else, save returndata to results array
                results[i] = ret;
            }
            unchecked {
                ++i;
            }
        }
    }

    // -------------------------------
    //   Common internal functions
    // -------------------------------

    /**
     * @dev Pulls `amount` of `token` from `user` into this contract.
     *      For ERC20: enforces `_msgSender() == user` (caller must have routed through `AllowanceHolder.exec`) and calls AH.transferFrom via assembly.
     *      AH selector: transferFrom(address,address,address,uint256) = 0x15dacbea.
     *      For native ETH: ETH must already be present as msg.value; verify sufficient value was forwarded.
     * @param token Input token or `CurrencyLib.NATIVE_TOKEN_ADDRESS`.
     * @param user Owner whose AllowanceHolder-scoped allowance is consumed.
     * @param amount Tokens or wei to pull.
     */
    function _pullFromUser(address token, address user, uint256 amount) internal {
        // Check input value if native token
        if (token == CurrencyLib.NATIVE_TOKEN_ADDRESS) {
            if (msg.value < amount) {
                revert InsufficientMsgValue();
            }
            return;
        }

        // Check caller is user
        if (_msgSender() != user) revert CallerNotSignedUser();

        // Call AllowanceHolder.transferFrom()
        address allowanceHolder = address(ALLOWANCE_HOLDER);
        assembly ("memory-safe") {
            // Manually ABI-encode AllowanceHolder.transferFrom(address token, address owner, address recipient, uint256 amount)
            // selector 0x15dacbea. Calldata is 0x84 (132) bytes and starts at ptr+0x1c (see last mstore below).
            //
            // The `shl(0x60, addr)` trick left-aligns a 20-byte address in a 32-byte word: the high 20 bytes
            // hold the address and the trailing 12 bytes are zero, which simultaneously encodes the address AND
            // provides the ABI zero-padding for the *next* field — so each shifted mstore clears the following
            // field's padding without a separate write.
            //
            // Calldata layout relative to ptr+0x1c:
            //   [0..3]    selector   (0x15dacbea)
            //   [4..35]   token      (12-byte pad + 20-byte address)
            //   [36..67]  owner/user (12-byte pad + 20-byte address)
            //   [68..99]  recipient  (12-byte pad + 20-byte address = address(this))
            //   [100..131] amount    (uint256)
            let ptr := mload(0x40)
            mstore(add(0x80, ptr), amount) // calldata[100..131]: amount (uint256, right-aligned)
            mstore(add(0x60, ptr), address()) // calldata[68..99]: recipient = this contract (right-aligned, high 12 bytes are zero padding)
            mstore(add(0x4c, ptr), shl(0x60, user)) // calldata[48..67]: user address; trailing 12 zero bytes fill calldata[68..79] (recipient padding)
            // `shl(0x60)` (96-bit), NOT `shl(0xa0)` (160-bit): 0xa0 here is literal 160, which
            // shifts the 20-byte address out of place and corrupts the calldata token. Same as 0x-settler `Permit2Payment._allowanceHolderTransferFrom`.
            mstore(add(0x2c, ptr), shl(0x60, token)) // calldata[16..35]: token address; trailing 12 zero bytes fill calldata[36..47] (user padding)
            mstore(add(0x0c, ptr), 0x15dacbea000000000000000000000000) // selector at calldata[0..3]; 12 zero bytes fill calldata[4..15] (token padding); calldata begins at ptr+0x1c

            if iszero(call(gas(), allowanceHolder, 0x00, add(0x1c, ptr), 0x84, 0x00, 0x00)) {
                // if call did not succeed, revert with the revert returndata
                let p := mload(0x40)
                returndatacopy(p, 0x00, returndatasize())
                revert(p, returndatasize())
            }
        }
    }

    /**
     * @dev Executes the swap call and returns the output amount.
     *      `useBalanceOf=true`: measure output as (balance after − balance before) at `outputReceiver`.
     *      `useBalanceOf=false`: decode output from returndata at `swapData.returnDataWordOffset`.
     *      `outputReceiver` must be `address(this)` when tokens are expected at the contract (post-swap fee path, bridge path)
     *      or the end user when the router sends directly to them.
     * @param swapData Swap target, value, output token, and returndata layout.
     * @param swapCallData Calldata forwarded to `swapData.target`.
     * @param useBalanceOf When true, use balance delta instead of returndata decoding.
     * @param outputReceiver Account whose output-token balance is measured or credited.
     * @return finalAmount Gross swap output amount.
     */
    function _execSwap(
        SwapData calldata swapData,
        bytes calldata swapCallData,
        bool useBalanceOf,
        address outputReceiver
    ) internal returns (uint256 finalAmount) {
        if (useBalanceOf) {
            // Measure output as (balance after − balance before) at `outputReceiver`
            uint256 before = CurrencyLib.balanceOf(swapData.outputToken, outputReceiver);
            _execCallCalldata(swapData.target, swapData.value, swapCallData, false);
            finalAmount = CurrencyLib.balanceOf(swapData.outputToken, outputReceiver) - before;
        } else {
            // Decode output from returndata
            bytes memory ret = _execCallCalldata(swapData.target, swapData.value, swapCallData, true);
            finalAmount = _decodeReturnWord(ret, swapData.returnDataWordOffset);
        }
    }

    /**
     * @dev Low-level `call` with bubbled revert data on failure.
     * @param target Call recipient.
     * @param value Wei forwarded with the call.
     * @param data ABI-encoded calldata in memory.
     */
    function _execCall(address target, uint256 value, bytes memory data) internal {
        bool success;
        assembly ("memory-safe") {
            success := call(gas(), target, value, add(data, 0x20), mload(data), 0, 0)
        }

        if (!success) {
            bytes memory ret;
            assembly ("memory-safe") {
                // prep and return revert data
                let returnDataSize := returndatasize()
                ret := mload(0x40)
                mstore(ret, returnDataSize) // write length prefix on free-mem pointer
                returndatacopy(add(ret, 0x20), 0, returnDataSize) // copy returndata after length
                mstore(0x40, and(add(add(add(ret, 0x20), returnDataSize), 0x1f), not(0x1f))) // bump free pointer
                revert(add(ret, 0x20), mload(ret)) // bubbles up the original revert payload
            }
        }
    }

    /**
     * @dev Low-level `call` using calldata copied to memory; optionally captures returndata.
     * @dev Helps cheaper external calls avoiding early copy of calldata to memory.
     * @param target Call recipient.
     * @param value Wei forwarded with the call.
     * @param data Calldata slice forwarded to `target`.
     * @param storeResult When true, copy returndata into memory even on success.
     * @return ret Returndata when `storeResult` is true or the call reverts (revert bubbles).
     */
    function _execCallCalldata(address target, uint256 value, bytes calldata data, bool storeResult)
        internal
        returns (bytes memory ret)
    {
        bool success;
        assembly ("memory-safe") {
            let ptr := mload(0x40)
            calldatacopy(ptr, data.offset, data.length) // copy calldata slice to fresh memory (avoids redundant memory alloc)
            mstore(0x40, and(add(add(ptr, data.length), 0x1f), not(0x1f))) // advance free pointer to next 32-byte boundary
            success := call(gas(), target, value, ptr, data.length, 0, 0)
        }

        if (!success || storeResult) {
            assembly ("memory-safe") {
                // prep and return revert data
                let returnDataSize := returndatasize()
                ret := mload(0x40)
                mstore(ret, returnDataSize) // write length prefix on free-mem pointer
                returndatacopy(add(ret, 0x20), 0, returnDataSize) // copy returndata after length
                mstore(0x40, and(add(add(add(ret, 0x20), returnDataSize), 0x1f), not(0x1f))) // bump free pointer
            }
            if (!success) {
                assembly ("memory-safe") {
                    revert(add(ret, 0x20), mload(ret)) // bubble up the raw revert payload
                }
            }
        }
    }

    /**
     * @dev Reads the 32-byte word at `wordOffset` from ABI-encoded `ret` (word index, not byte offset).
     * @param ret Return blob from a prior call.
     * @param wordOffset Zero-based index of the 32-byte word to load.
     * @return word Decoded amount or value at that offset.
     */
    function _decodeReturnWord(bytes memory ret, uint256 wordOffset) internal pure returns (uint256 word) {
        uint256 offset = wordOffset * 32;
        if (offset + 32 > ret.length) revert ReturnDataOutOfBounds();

        assembly ("memory-safe") {
            // read the word at the offset from return data
            word := mload(add(add(ret, 0x20), offset))
        }
    }

    /**
     * @notice Rescues funds from the contract if they are locked by mistake.
     * @param token The address of the token contract.
     * @param rescueTo The address where rescued tokens need to be sent.
     * @param amount The amount of tokens to be rescued.
     */
    function rescueFunds(address token, address rescueTo, uint256 amount) external onlyRole(RESCUE_ROLE) {
        RescueFundsLib.rescueFunds(token, rescueTo, amount);
    }
}


// ===== FILE: lib/solady/src/utils/SafeTransferLib.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

/// @notice Safe ETH and ERC20 transfer library that gracefully handles missing return values.
/// @author Solady (https://github.com/vectorized/solady/blob/main/src/utils/SafeTransferLib.sol)
/// @author Modified from Solmate (https://github.com/transmissions11/solmate/blob/main/src/utils/SafeTransferLib.sol)
/// @author Permit2 operations from (https://github.com/Uniswap/permit2/blob/main/src/libraries/Permit2Lib.sol)
///
/// @dev Note:
/// - For ETH transfers, please use `forceSafeTransferETH` for DoS protection.
library SafeTransferLib {
    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                       CUSTOM ERRORS                        */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev The ETH transfer has failed.
    error ETHTransferFailed();

    /// @dev The ERC20 `transferFrom` has failed.
    error TransferFromFailed();

    /// @dev The ERC20 `transfer` has failed.
    error TransferFailed();

    /// @dev The ERC20 `approve` has failed.
    error ApproveFailed();

    /// @dev The ERC20 `totalSupply` query has failed.
    error TotalSupplyQueryFailed();

    /// @dev The Permit2 operation has failed.
    error Permit2Failed();

    /// @dev The Permit2 amount must be less than `2**160 - 1`.
    error Permit2AmountOverflow();

    /// @dev The Permit2 approve operation has failed.
    error Permit2ApproveFailed();

    /// @dev The Permit2 lockdown operation has failed.
    error Permit2LockdownFailed();

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                         CONSTANTS                          */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Suggested gas stipend for contract receiving ETH that disallows any storage writes.
    uint256 internal constant GAS_STIPEND_NO_STORAGE_WRITES = 2300;

    /// @dev Suggested gas stipend for contract receiving ETH to perform a few
    /// storage reads and writes, but low enough to prevent griefing.
    uint256 internal constant GAS_STIPEND_NO_GRIEF = 100000;

    /// @dev The unique EIP-712 domain separator for the DAI token contract.
    bytes32 internal constant DAI_DOMAIN_SEPARATOR =
        0xdbb8cf42e1ecb028be3f3dbc922e1d878b963f411dc388ced501601c60f7c6f7;

    /// @dev The address for the WETH9 contract on Ethereum mainnet.
    address internal constant WETH9 = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;

    /// @dev The canonical Permit2 address.
    /// [Github](https://github.com/Uniswap/permit2)
    /// [Etherscan](https://etherscan.io/address/0x000000000022D473030F116dDEE9F6B43aC78BA3)
    address internal constant PERMIT2 = 0x000000000022D473030F116dDEE9F6B43aC78BA3;

    /// @dev The canonical address of the `SELFDESTRUCT` ETH mover.
    /// See: https://gist.github.com/Vectorized/1cb8ad4cf393b1378e08f23f79bd99fa
    /// [Etherscan](https://etherscan.io/address/0x00000000000073c48c8055bD43D1A53799176f0D)
    address internal constant ETH_MOVER = 0x00000000000073c48c8055bD43D1A53799176f0D;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                       ETH OPERATIONS                       */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    // If the ETH transfer MUST succeed with a reasonable gas budget, use the force variants.
    //
    // The regular variants:
    // - Forwards all remaining gas to the target.
    // - Reverts if the target reverts.
    // - Reverts if the current contract has insufficient balance.
    //
    // The force variants:
    // - Forwards with an optional gas stipend
    //   (defaults to `GAS_STIPEND_NO_GRIEF`, which is sufficient for most cases).
    // - If the target reverts, or if the gas stipend is exhausted,
    //   creates a temporary contract to force send the ETH via `SELFDESTRUCT`.
    //   Future compatible with `SENDALL`: https://eips.ethereum.org/EIPS/eip-4758.
    // - Reverts if the current contract has insufficient balance.
    //
    // The try variants:
    // - Forwards with a mandatory gas stipend.
    // - Instead of reverting, returns whether the transfer succeeded.

    /// @dev Sends `amount` (in wei) ETH to `to`.
    function safeTransferETH(address to, uint256 amount) internal {
        /// @solidity memory-safe-assembly
        assembly {
            if iszero(call(gas(), to, amount, codesize(), 0x00, codesize(), 0x00)) {
                mstore(0x00, 0xb12d13eb) // `ETHTransferFailed()`.
                revert(0x1c, 0x04)
            }
        }
    }

    /// @dev Sends all the ETH in the current contract to `to`.
    function safeTransferAllETH(address to) internal {
        /// @solidity memory-safe-assembly
        assembly {
            // Transfer all the ETH and check if it succeeded or not.
            if iszero(call(gas(), to, selfbalance(), codesize(), 0x00, codesize(), 0x00)) {
                mstore(0x00, 0xb12d13eb) // `ETHTransferFailed()`.
                revert(0x1c, 0x04)
            }
        }
    }

    /// @dev Force sends `amount` (in wei) ETH to `to`, with a `gasStipend`.
    function forceSafeTransferETH(address to, uint256 amount, uint256 gasStipend) internal {
        /// @solidity memory-safe-assembly
        assembly {
            if lt(selfbalance(), amount) {
                mstore(0x00, 0xb12d13eb) // `ETHTransferFailed()`.
                revert(0x1c, 0x04)
            }
            if iszero(call(gasStipend, to, amount, codesize(), 0x00, codesize(), 0x00)) {
                mstore(0x00, to) // Store the address in scratch space.
                mstore8(0x0b, 0x73) // Opcode `PUSH20`.
                mstore8(0x20, 0xff) // Opcode `SELFDESTRUCT`.
                if iszero(create(amount, 0x0b, 0x16)) { revert(codesize(), codesize()) } // For gas estimation.
            }
        }
    }

    /// @dev Force sends all the ETH in the current contract to `to`, with a `gasStipend`.
    function forceSafeTransferAllETH(address to, uint256 gasStipend) internal {
        /// @solidity memory-safe-assembly
        assembly {
            if iszero(call(gasStipend, to, selfbalance(), codesize(), 0x00, codesize(), 0x00)) {
                mstore(0x00, to) // Store the address in scratch space.
                mstore8(0x0b, 0x73) // Opcode `PUSH20`.
                mstore8(0x20, 0xff) // Opcode `SELFDESTRUCT`.
                if iszero(create(selfbalance(), 0x0b, 0x16)) { revert(codesize(), codesize()) } // For gas estimation.
            }
        }
    }

    /// @dev Force sends `amount` (in wei) ETH to `to`, with `GAS_STIPEND_NO_GRIEF`.
    function forceSafeTransferETH(address to, uint256 amount) internal {
        /// @solidity memory-safe-assembly
        assembly {
            if lt(selfbalance(), amount) {
                mstore(0x00, 0xb12d13eb) // `ETHTransferFailed()`.
                revert(0x1c, 0x04)
            }
            if iszero(call(GAS_STIPEND_NO_GRIEF, to, amount, codesize(), 0x00, codesize(), 0x00)) {
                mstore(0x00, to) // Store the address in scratch space.
                mstore8(0x0b, 0x73) // Opcode `PUSH20`.
                mstore8(0x20, 0xff) // Opcode `SELFDESTRUCT`.
                if iszero(create(amount, 0x0b, 0x16)) { revert(codesize(), codesize()) } // For gas estimation.
            }
        }
    }

    /// @dev Force sends all the ETH in the current contract to `to`, with `GAS_STIPEND_NO_GRIEF`.
    function forceSafeTransferAllETH(address to) internal {
        /// @solidity memory-safe-assembly
        assembly {
            // forgefmt: disable-next-item
            if iszero(call(GAS_STIPEND_NO_GRIEF, to, selfbalance(), codesize(), 0x00, codesize(), 0x00)) {
                mstore(0x00, to) // Store the address in scratch space.
                mstore8(0x0b, 0x73) // Opcode `PUSH20`.
                mstore8(0x20, 0xff) // Opcode `SELFDESTRUCT`.
                if iszero(create(selfbalance(), 0x0b, 0x16)) { revert(codesize(), codesize()) } // For gas estimation.
            }
        }
    }

    /// @dev Sends `amount` (in wei) ETH to `to`, with a `gasStipend`.
    function trySafeTransferETH(address to, uint256 amount, uint256 gasStipend)
        internal
        returns (bool success)
    {
        /// @solidity memory-safe-assembly
        assembly {
            success := call(gasStipend, to, amount, codesize(), 0x00, codesize(), 0x00)
        }
    }

    /// @dev Sends all the ETH in the current contract to `to`, with a `gasStipend`.
    function trySafeTransferAllETH(address to, uint256 gasStipend)
        internal
        returns (bool success)
    {
        /// @solidity memory-safe-assembly
        assembly {
            success := call(gasStipend, to, selfbalance(), codesize(), 0x00, codesize(), 0x00)
        }
    }

    /// @dev Force transfers ETH to `to`, without triggering the fallback (if any).
    /// This method attempts to use a separate contract to send via `SELFDESTRUCT`,
    /// and upon failure, deploys a minimal vault to accrue the ETH.
    function safeMoveETH(address to, uint256 amount) internal returns (address vault) {
        /// @solidity memory-safe-assembly
        assembly {
            to := shr(96, shl(96, to)) // Clean upper 96 bits.
            for { let mover := ETH_MOVER } iszero(eq(to, address())) {} {
                let selfBalanceBefore := selfbalance()
                if or(lt(selfBalanceBefore, amount), eq(to, mover)) {
                    mstore(0x00, 0xb12d13eb) // `ETHTransferFailed()`.
                    revert(0x1c, 0x04)
                }
                if extcodesize(mover) {
                    let balanceBefore := balance(to) // Check via delta, in case `SELFDESTRUCT` is bricked.
                    mstore(0x00, to)
                    pop(call(gas(), mover, amount, 0x00, 0x20, codesize(), 0x00))
                    // If `address(to).balance >= amount + balanceBefore`, skip vault workflow.
                    if iszero(lt(balance(to), add(amount, balanceBefore))) { break }
                    // Just in case `SELFDESTRUCT` is changed to not revert and do nothing.
                    if lt(selfBalanceBefore, selfbalance()) { invalid() }
                }
                let m := mload(0x40)
                // If the mover is missing or bricked, deploy a minimal vault
                // that withdraws all ETH to `to` when being called only by `to`.
                // forgefmt: disable-next-item
                mstore(add(m, 0x20), 0x33146025575b600160005260206000f35b3d3d3d3d47335af1601a5760003dfd)
                mstore(m, or(to, shl(160, 0x6035600b3d3960353df3fe73)))
                // Compute and store the bytecode hash.
                mstore8(0x00, 0xff) // Write the prefix.
                mstore(0x35, keccak256(m, 0x40))
                mstore(0x01, shl(96, address())) // Deployer.
                mstore(0x15, 0) // Salt.
                vault := keccak256(0x00, 0x55)
                pop(call(gas(), vault, amount, codesize(), 0x00, codesize(), 0x00))
                // The vault returns a single word on success. Failure reverts with empty data.
                if iszero(returndatasize()) {
                    if iszero(create2(0, m, 0x40, 0)) { revert(codesize(), codesize()) } // For gas estimation.
                }
                mstore(0x40, m) // Restore the free memory pointer.
                break
            }
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                      ERC20 OPERATIONS                      */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Sends `amount` of ERC20 `token` from `from` to `to`.
    /// Reverts upon failure.
    ///
    /// The `from` account must have at least `amount` approved for
    /// the current contract to manage.
    function safeTransferFrom(address token, address from, address to, uint256 amount) internal {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40) // Cache the free memory pointer.
            mstore(0x60, amount) // Store the `amount` argument.
            mstore(0x40, to) // Store the `to` argument.
            mstore(0x2c, shl(96, from)) // Store the `from` argument.
            mstore(0x0c, 0x23b872dd000000000000000000000000) // `transferFrom(address,address,uint256)`.
            let success := call(gas(), token, 0, 0x1c, 0x64, 0x00, 0x20)
            if iszero(and(eq(mload(0x00), 1), success)) {
                if iszero(lt(or(iszero(extcodesize(token)), returndatasize()), success)) {
                    mstore(0x00, 0x7939f424) // `TransferFromFailed()`.
                    revert(0x1c, 0x04)
                }
            }
            mstore(0x60, 0) // Restore the zero slot to zero.
            mstore(0x40, m) // Restore the free memory pointer.
        }
    }

    /// @dev Sends `amount` of ERC20 `token` from `from` to `to`.
    ///
    /// The `from` account must have at least `amount` approved for the current contract to manage.
    function trySafeTransferFrom(address token, address from, address to, uint256 amount)
        internal
        returns (bool success)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40) // Cache the free memory pointer.
            mstore(0x60, amount) // Store the `amount` argument.
            mstore(0x40, to) // Store the `to` argument.
            mstore(0x2c, shl(96, from)) // Store the `from` argument.
            mstore(0x0c, 0x23b872dd000000000000000000000000) // `transferFrom(address,address,uint256)`.
            success := call(gas(), token, 0, 0x1c, 0x64, 0x00, 0x20)
            if iszero(and(eq(mload(0x00), 1), success)) {
                success := lt(or(iszero(extcodesize(token)), returndatasize()), success)
            }
            mstore(0x60, 0) // Restore the zero slot to zero.
            mstore(0x40, m) // Restore the free memory pointer.
        }
    }

    /// @dev Sends all of ERC20 `token` from `from` to `to`.
    /// Reverts upon failure.
    ///
    /// The `from` account must have their entire balance approved for the current contract to manage.
    function safeTransferAllFrom(address token, address from, address to)
        internal
        returns (uint256 amount)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40) // Cache the free memory pointer.
            mstore(0x40, to) // Store the `to` argument.
            mstore(0x2c, shl(96, from)) // Store the `from` argument.
            mstore(0x0c, 0x70a08231000000000000000000000000) // `balanceOf(address)`.
            // Read the balance, reverting upon failure.
            if iszero(
                and( // The arguments of `and` are evaluated from right to left.
                    gt(returndatasize(), 0x1f), // At least 32 bytes returned.
                    staticcall(gas(), token, 0x1c, 0x24, 0x60, 0x20)
                )
            ) {
                mstore(0x00, 0x7939f424) // `TransferFromFailed()`.
                revert(0x1c, 0x04)
            }
            mstore(0x00, 0x23b872dd) // `transferFrom(address,address,uint256)`.
            amount := mload(0x60) // The `amount` is already at 0x60. We'll need to return it.
            // Perform the transfer, reverting upon failure.
            let success := call(gas(), token, 0, 0x1c, 0x64, 0x00, 0x20)
            if iszero(and(eq(mload(0x00), 1), success)) {
                if iszero(lt(or(iszero(extcodesize(token)), returndatasize()), success)) {
                    mstore(0x00, 0x7939f424) // `TransferFromFailed()`.
                    revert(0x1c, 0x04)
                }
            }
            mstore(0x60, 0) // Restore the zero slot to zero.
            mstore(0x40, m) // Restore the free memory pointer.
        }
    }

    /// @dev Sends `amount` of ERC20 `token` from the current contract to `to`.
    /// Reverts upon failure.
    function safeTransfer(address token, address to, uint256 amount) internal {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x14, to) // Store the `to` argument.
            mstore(0x34, amount) // Store the `amount` argument.
            mstore(0x00, 0xa9059cbb000000000000000000000000) // `transfer(address,uint256)`.
            // Perform the transfer, reverting upon failure.
            let success := call(gas(), token, 0, 0x10, 0x44, 0x00, 0x20)
            if iszero(and(eq(mload(0x00), 1), success)) {
                if iszero(lt(or(iszero(extcodesize(token)), returndatasize()), success)) {
                    mstore(0x00, 0x90b8ec18) // `TransferFailed()`.
                    revert(0x1c, 0x04)
                }
            }
            mstore(0x34, 0) // Restore the part of the free memory pointer that was overwritten.
        }
    }

    /// @dev Sends all of ERC20 `token` from the current contract to `to`.
    /// Reverts upon failure.
    function safeTransferAll(address token, address to) internal returns (uint256 amount) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, 0x70a08231) // Store the function selector of `balanceOf(address)`.
            mstore(0x20, address()) // Store the address of the current contract.
            // Read the balance, reverting upon failure.
            if iszero(
                and( // The arguments of `and` are evaluated from right to left.
                    gt(returndatasize(), 0x1f), // At least 32 bytes returned.
                    staticcall(gas(), token, 0x1c, 0x24, 0x34, 0x20)
                )
            ) {
                mstore(0x00, 0x90b8ec18) // `TransferFailed()`.
                revert(0x1c, 0x04)
            }
            mstore(0x14, to) // Store the `to` argument.
            amount := mload(0x34) // The `amount` is already at 0x34. We'll need to return it.
            mstore(0x00, 0xa9059cbb000000000000000000000000) // `transfer(address,uint256)`.
            // Perform the transfer, reverting upon failure.
            let success := call(gas(), token, 0, 0x10, 0x44, 0x00, 0x20)
            if iszero(and(eq(mload(0x00), 1), success)) {
                if iszero(lt(or(iszero(extcodesize(token)), returndatasize()), success)) {
                    mstore(0x00, 0x90b8ec18) // `TransferFailed()`.
                    revert(0x1c, 0x04)
                }
            }
            mstore(0x34, 0) // Restore the part of the free memory pointer that was overwritten.
        }
    }

    /// @dev Sets `amount` of ERC20 `token` for `to` to manage on behalf of the current contract.
    /// Reverts upon failure.
    function safeApprove(address token, address to, uint256 amount) internal {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x14, to) // Store the `to` argument.
            mstore(0x34, amount) // Store the `amount` argument.
            mstore(0x00, 0x095ea7b3000000000000000000000000) // `approve(address,uint256)`.
            let success := call(gas(), token, 0, 0x10, 0x44, 0x00, 0x20)
            if iszero(and(eq(mload(0x00), 1), success)) {
                if iszero(lt(or(iszero(extcodesize(token)), returndatasize()), success)) {
                    mstore(0x00, 0x3e3f8f73) // `ApproveFailed()`.
                    revert(0x1c, 0x04)
                }
            }
            mstore(0x34, 0) // Restore the part of the free memory pointer that was overwritten.
        }
    }

    /// @dev Sets `amount` of ERC20 `token` for `to` to manage on behalf of the current contract.
    /// If the initial attempt to approve fails, attempts to reset the approved amount to zero,
    /// then retries the approval again (some tokens, e.g. USDT, requires this).
    /// Reverts upon failure.
    function safeApproveWithRetry(address token, address to, uint256 amount) internal {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x14, to) // Store the `to` argument.
            mstore(0x34, amount) // Store the `amount` argument.
            mstore(0x00, 0x095ea7b3000000000000000000000000) // `approve(address,uint256)`.
            // Perform the approval, retrying upon failure.
            let success := call(gas(), token, 0, 0x10, 0x44, 0x00, 0x20)
            if iszero(and(eq(mload(0x00), 1), success)) {
                if iszero(lt(or(iszero(extcodesize(token)), returndatasize()), success)) {
                    mstore(0x34, 0) // Store 0 for the `amount`.
                    mstore(0x00, 0x095ea7b3000000000000000000000000) // `approve(address,uint256)`.
                    pop(call(gas(), token, 0, 0x10, 0x44, codesize(), 0x00)) // Reset the approval.
                    mstore(0x34, amount) // Store back the original `amount`.
                    // Retry the approval, reverting upon failure.
                    success := call(gas(), token, 0, 0x10, 0x44, 0x00, 0x20)
                    if iszero(and(eq(mload(0x00), 1), success)) {
                        // Check the `extcodesize` again just in case the token selfdestructs lol.
                        if iszero(lt(or(iszero(extcodesize(token)), returndatasize()), success)) {
                            mstore(0x00, 0x3e3f8f73) // `ApproveFailed()`.
                            revert(0x1c, 0x04)
                        }
                    }
                }
            }
            mstore(0x34, 0) // Restore the part of the free memory pointer that was overwritten.
        }
    }

    /// @dev Returns the amount of ERC20 `token` owned by `account`.
    /// Returns zero if the `token` does not exist.
    function balanceOf(address token, address account) internal view returns (uint256 amount) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x14, account) // Store the `account` argument.
            mstore(0x00, 0x70a08231000000000000000000000000) // `balanceOf(address)`.
            amount :=
                mul( // The arguments of `mul` are evaluated from right to left.
                    mload(0x20),
                    and( // The arguments of `and` are evaluated from right to left.
                        gt(returndatasize(), 0x1f), // At least 32 bytes returned.
                        staticcall(gas(), token, 0x10, 0x24, 0x20, 0x20)
                    )
                )
        }
    }

    /// @dev Performs a `token.balanceOf(account)` check.
    /// `implemented` denotes whether the `token` does not implement `balanceOf`.
    /// `amount` is zero if the `token` does not implement `balanceOf`.
    function checkBalanceOf(address token, address account)
        internal
        view
        returns (bool implemented, uint256 amount)
    {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x14, account) // Store the `account` argument.
            mstore(0x00, 0x70a08231000000000000000000000000) // `balanceOf(address)`.
            implemented :=
                and( // The arguments of `and` are evaluated from right to left.
                    gt(returndatasize(), 0x1f), // At least 32 bytes returned.
                    staticcall(gas(), token, 0x10, 0x24, 0x20, 0x20)
                )
            amount := mul(mload(0x20), implemented)
        }
    }

    /// @dev Returns the total supply of the `token`.
    /// Reverts if the token does not exist or does not implement `totalSupply()`.
    function totalSupply(address token) internal view returns (uint256 result) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, 0x18160ddd) // `totalSupply()`.
            if iszero(
                and(gt(returndatasize(), 0x1f), staticcall(gas(), token, 0x1c, 0x04, 0x00, 0x20))
            ) {
                mstore(0x00, 0x54cd9435) // `TotalSupplyQueryFailed()`.
                revert(0x1c, 0x04)
            }
            result := mload(0x00)
        }
    }

    /// @dev Sends `amount` of ERC20 `token` from `from` to `to`.
    /// If the initial attempt fails, try to use Permit2 to transfer the token.
    /// Reverts upon failure.
    ///
    /// The `from` account must have at least `amount` approved for the current contract to manage.
    function safeTransferFrom2(address token, address from, address to, uint256 amount) internal {
        if (!trySafeTransferFrom(token, from, to, amount)) {
            permit2TransferFrom(token, from, to, amount);
        }
    }

    /// @dev Sends `amount` of ERC20 `token` from `from` to `to` via Permit2.
    /// Reverts upon failure.
    function permit2TransferFrom(address token, address from, address to, uint256 amount)
        internal
    {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(add(m, 0x74), shr(96, shl(96, token)))
            mstore(add(m, 0x54), amount)
            mstore(add(m, 0x34), to)
            mstore(add(m, 0x20), shl(96, from))
            // `transferFrom(address,address,uint160,address)`.
            mstore(m, 0x36c78516000000000000000000000000)
            let p := PERMIT2
            let exists := eq(chainid(), 1)
            if iszero(exists) { exists := iszero(iszero(extcodesize(p))) }
            if iszero(
                and(
                    call(gas(), p, 0, add(m, 0x10), 0x84, codesize(), 0x00),
                    lt(iszero(extcodesize(token)), exists) // Token has code and Permit2 exists.
                )
            ) {
                mstore(0x00, 0x7939f4248757f0fd) // `TransferFromFailed()` or `Permit2AmountOverflow()`.
                revert(add(0x18, shl(2, iszero(iszero(shr(160, amount))))), 0x04)
            }
        }
    }

    /// @dev Permit a user to spend a given amount of
    /// another user's tokens via native EIP-2612 permit if possible, falling
    /// back to Permit2 if native permit fails or is not implemented on the token.
    function permit2(
        address token,
        address owner,
        address spender,
        uint256 amount,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) internal {
        bool success;
        /// @solidity memory-safe-assembly
        assembly {
            for {} shl(96, xor(token, WETH9)) {} {
                mstore(0x00, 0x3644e515) // `DOMAIN_SEPARATOR()`.
                if iszero(
                    and( // The arguments of `and` are evaluated from right to left.
                        lt(iszero(mload(0x00)), eq(returndatasize(), 0x20)), // Returns 1 non-zero word.
                        // Gas stipend to limit gas burn for tokens that don't refund gas when
                        // an non-existing function is called. 5K should be enough for a SLOAD.
                        staticcall(5000, token, 0x1c, 0x04, 0x00, 0x20)
                    )
                ) { break }
                // After here, we can be sure that token is a contract.
                let m := mload(0x40)
                mstore(add(m, 0x34), spender)
                mstore(add(m, 0x20), shl(96, owner))
                mstore(add(m, 0x74), deadline)
                if eq(mload(0x00), DAI_DOMAIN_SEPARATOR) {
                    mstore(0x14, owner)
                    mstore(0x00, 0x7ecebe00000000000000000000000000) // `nonces(address)`.
                    mstore(
                        add(m, 0x94),
                        lt(iszero(amount), staticcall(gas(), token, 0x10, 0x24, add(m, 0x54), 0x20))
                    )
                    mstore(m, 0x8fcbaf0c000000000000000000000000) // `IDAIPermit.permit`.
                    // `nonces` is already at `add(m, 0x54)`.
                    // `amount != 0` is already stored at `add(m, 0x94)`.
                    mstore(add(m, 0xb4), and(0xff, v))
                    mstore(add(m, 0xd4), r)
                    mstore(add(m, 0xf4), s)
                    success := call(gas(), token, 0, add(m, 0x10), 0x104, codesize(), 0x00)
                    break
                }
                mstore(m, 0xd505accf000000000000000000000000) // `IERC20Permit.permit`.
                mstore(add(m, 0x54), amount)
                mstore(add(m, 0x94), and(0xff, v))
                mstore(add(m, 0xb4), r)
                mstore(add(m, 0xd4), s)
                success := call(gas(), token, 0, add(m, 0x10), 0xe4, codesize(), 0x00)
                break
            }
        }
        if (!success) simplePermit2(token, owner, spender, amount, deadline, v, r, s);
    }

    /// @dev Simple permit on the Permit2 contract.
    function simplePermit2(
        address token,
        address owner,
        address spender,
        uint256 amount,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) internal {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, 0x927da105) // `allowance(address,address,address)`.
            {
                let addressMask := shr(96, not(0))
                mstore(add(m, 0x20), and(addressMask, owner))
                mstore(add(m, 0x40), and(addressMask, token))
                mstore(add(m, 0x60), and(addressMask, spender))
                mstore(add(m, 0xc0), and(addressMask, spender))
            }
            let p := mul(PERMIT2, iszero(shr(160, amount)))
            if iszero(
                and( // The arguments of `and` are evaluated from right to left.
                    gt(returndatasize(), 0x5f), // Returns 3 words: `amount`, `expiration`, `nonce`.
                    staticcall(gas(), p, add(m, 0x1c), 0x64, add(m, 0x60), 0x60)
                )
            ) {
                mstore(0x00, 0x6b836e6b8757f0fd) // `Permit2Failed()` or `Permit2AmountOverflow()`.
                revert(add(0x18, shl(2, iszero(p))), 0x04)
            }
            mstore(m, 0x2b67b570) // `Permit2.permit` (PermitSingle variant).
            // `owner` is already `add(m, 0x20)`.
            // `token` is already at `add(m, 0x40)`.
            mstore(add(m, 0x60), amount)
            mstore(add(m, 0x80), 0xffffffffffff) // `expiration = type(uint48).max`.
            // `nonce` is already at `add(m, 0xa0)`.
            // `spender` is already at `add(m, 0xc0)`.
            mstore(add(m, 0xe0), deadline)
            mstore(add(m, 0x100), 0x100) // `signature` offset.
            mstore(add(m, 0x120), 0x41) // `signature` length.
            mstore(add(m, 0x140), r)
            mstore(add(m, 0x160), s)
            mstore(add(m, 0x180), shl(248, v))
            if iszero( // Revert if token does not have code, or if the call fails.
            mul(extcodesize(token), call(gas(), p, 0, add(m, 0x1c), 0x184, codesize(), 0x00))) {
                mstore(0x00, 0x6b836e6b) // `Permit2Failed()`.
                revert(0x1c, 0x04)
            }
        }
    }

    /// @dev Approves `spender` to spend `amount` of `token` for `address(this)`.
    function permit2Approve(address token, address spender, uint160 amount, uint48 expiration)
        internal
    {
        /// @solidity memory-safe-assembly
        assembly {
            let addressMask := shr(96, not(0))
            let m := mload(0x40)
            mstore(m, 0x87517c45) // `approve(address,address,uint160,uint48)`.
            mstore(add(m, 0x20), and(addressMask, token))
            mstore(add(m, 0x40), and(addressMask, spender))
            mstore(add(m, 0x60), and(addressMask, amount))
            mstore(add(m, 0x80), and(0xffffffffffff, expiration))
            if iszero(call(gas(), PERMIT2, 0, add(m, 0x1c), 0xa0, codesize(), 0x00)) {
                mstore(0x00, 0x324f14ae) // `Permit2ApproveFailed()`.
                revert(0x1c, 0x04)
            }
        }
    }

    /// @dev Revokes an approval for `token` and `spender` for `address(this)`.
    function permit2Lockdown(address token, address spender) internal {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, 0xcc53287f) // `Permit2.lockdown`.
            mstore(add(m, 0x20), 0x20) // Offset of the `approvals`.
            mstore(add(m, 0x40), 1) // `approvals.length`.
            mstore(add(m, 0x60), shr(96, shl(96, token)))
            mstore(add(m, 0x80), shr(96, shl(96, spender)))
            if iszero(call(gas(), PERMIT2, 0, add(m, 0x1c), 0xa0, codesize(), 0x00)) {
                mstore(0x00, 0x96b3de23) // `Permit2LockdownFailed()`.
                revert(0x1c, 0x04)
            }
        }
    }
}


// ===== FILE: src/common/interfaces/IERC20.sol =====
// SPDX-License-Identifier: GPL-3.0-only
pragma solidity 0.8.34;

interface IERC20 {
    function allowance(address owner, address spender) external view returns (uint256);
}


// ===== FILE: src/common/utils/AccessControl.sol =====
// SPDX-License-Identifier: GPL-3.0-only
pragma solidity 0.8.34;

import {Ownable} from "./Ownable.sol";

// @audit Audited before by Zellic: https://github.com/SocketDotTech/audits/blob/main/Socket-DL/07-2023%20-%20Data%20Layer%20-%20Zellic.pdf
abstract contract AccessControl is Ownable {
    mapping(bytes32 => mapping(address => bool)) private _permits;

    event RoleGranted(bytes32 indexed role, address indexed grantee);
    event RoleRevoked(bytes32 indexed role, address indexed revokee);

    error NoPermit(bytes32 role);

    constructor(address owner_) Ownable(owner_) {}

    modifier onlyRole(bytes32 role) {
        if (!_permits[role][msg.sender]) revert NoPermit(role);
        _;
    }

    function grantRole(bytes32 role_, address grantee_) external virtual onlyOwner {
        _grantRole(role_, grantee_);
    }

    function revokeRole(bytes32 role_, address revokee_) external virtual onlyOwner {
        _revokeRole(role_, revokee_);
    }

    function hasRole(bytes32 role_, address address_) public view returns (bool) {
        return _hasRole(role_, address_);
    }

    function _grantRole(bytes32 role_, address grantee_) internal {
        _permits[role_][grantee_] = true;
        emit RoleGranted(role_, grantee_);
    }

    function _revokeRole(bytes32 role_, address revokee_) internal {
        _permits[role_][revokee_] = false;
        emit RoleRevoked(role_, revokee_);
    }

    function _hasRole(bytes32 role_, address address_) internal view returns (bool) {
        return _permits[role_][address_];
    }
}


// ===== FILE: src/common/allowance/AllowanceHolderContext.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import {ALLOWANCE_HOLDER} from "../interfaces/IAllowanceHolder.sol";

/// @title AllowanceHolderContext
/// @notice Self-contained port of 0x-settler's AllowanceHolderContext. Provides
///         `_msgSender()` that returns the original user when this contract is
///         called via `AllowanceHolder.exec(...)` (which appends the user's
///         address as the last 20 bytes of calldata, ERC-2771 style).
///         Also exposes the dummy `balanceOf(address)` so AllowanceHolder's
///         confused-deputy probe (it tries to call `balanceOf` on the target
///         to detect whether `target` looks like an ERC20) does not reject the
///         call.
abstract contract AllowanceHolderContext {
    /// @notice Returns the effective msg.sender. When the immediate caller is
    ///         `AllowanceHolder`, the trusted forwarder, the real user is
    ///         decoded from the last 20 bytes of calldata.
    function _msgSender() internal view virtual returns (address sender) {
        sender = msg.sender;
        if (sender == address(ALLOWANCE_HOLDER)) {
            // ERC-2771 style: AllowanceHolder appends the user's address as
            // the trailing 20 bytes of calldata after invoking target.
            assembly ("memory-safe") {
                sender := shr(0x60, calldataload(sub(calldatasize(), 0x14)))
            }
        }
    }

    /// @notice True if the call was forwarded by AllowanceHolder.
    function _isForwarded() internal view virtual returns (bool) {
        return msg.sender == address(ALLOWANCE_HOLDER);
    }

    /// @notice msg.data with the trailing 20-byte user address stripped when
    ///         the call was forwarded by AllowanceHolder.
    function _msgData() internal view virtual returns (bytes calldata) {
        if (msg.sender == address(ALLOWANCE_HOLDER)) {
            return msg.data[:msg.data.length - 20];
        }
        return msg.data;
    }

    /// @notice Dummy `balanceOf` implementation. AllowanceHolder probes its
    ///         `target` with `IERC20.balanceOf(...)` and reverts if the call
    ///         returns 32 bytes (i.e. target looks like an ERC20). We return
    ///         a single-byte return so the check passes, mirroring 0x-settler.
    function balanceOf(address) external pure {
        assembly ("memory-safe") {
            mstore8(0x00, 0x00)
            return(0x00, 0x01)
        }
    }
}


// ===== FILE: src/common/interfaces/IAllowanceHolder.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

// @dev Mainnet AllowanceHolder address. Same address is used for every chain
//      on which 0x deploys it via the canonical CREATE2 deployer. See:
//      https://docs.0x.org/docs/core-concepts/contracts#allowanceholder-recommended
IAllowanceHolder constant ALLOWANCE_HOLDER = IAllowanceHolder(0x0000000000001fF3684f28c67538d4D072C22734);

/// @title IAllowanceHolder
/// @notice External-facing interface of 0x's AllowanceHolder contract.
///         Mirrors `0x-settler/src/allowanceholder/IAllowanceHolder.sol`.
interface IAllowanceHolder {
    /// @notice The user calls `exec(operator, token, amount, target, data)` on
    ///         AllowanceHolder. AllowanceHolder writes a transient allowance for
    ///         `(operator, msgSender, token)` of `amount`, then calls `target`
    ///         with `data` and the user's address appended ERC-2771-style.
    function exec(address operator, address token, uint256 amount, address payable target, bytes calldata data)
        external
        payable
        returns (bytes memory result);

    /// @notice Counterpart to `exec`. Called by `operator` (the OpenRouter)
    ///         to consume the transient allowance and pull `amount` of
    ///         `token` from `owner` to `recipient`.
    function transferFrom(address token, address owner, address recipient, uint256 amount) external returns (bool);
}


// ===== FILE: src/common/lib/BytesSpliceLib.sol =====
// SPDX-License-Identifier: GPL-3.0-only
pragma solidity 0.8.34;

/// @title BytesSpliceLib
/// @notice Generalisation of the in-place calldata patching. Supports patching
///         either a single 32-byte word (for `uint256` amount fields) or an
///         arbitrary length copy from one bytes blob to another.
library BytesSpliceLib {
    error SpliceLengthZero();
    error SpliceSrcOutOfBounds();
    error SpliceDstOutOfBounds();
    error SplicePositionOutOfBounds();

    /// @notice Overwrites a 32-byte word at `position` in `data` with `word`.
    function spliceWord(bytes memory data, uint256 position, uint256 word) internal pure {
        // Bounds check: position + 32 must fit in data
        if (position + 32 > data.length) {
            revert SplicePositionOutOfBounds();
        }
        // in-place mstore at data + 32 (skip length prefix) + position
        assembly ("memory-safe") {
            mstore(add(add(data, 0x20), position), word)
        }
    }

    /// @notice Overwrites a 32-byte word at every entry in `positions`.
    function spliceWords(bytes memory data, uint256[] memory positions, uint256 word) internal pure {
        uint256 len = positions.length;
        for (uint256 i = 0; i < len;) {
            spliceWord({data: data, position: positions[i], word: word});
            unchecked {
                ++i;
            }
        }
    }

    /// @notice Copies `length` bytes from `src` at `srcOffset` into `dst` at `dstOffset`.
    /// @dev Uses Cancun's `mcopy`. Performs bounds checks on both blobs and
    ///      rejects zero-length splices to match the safety expectations on
    ///      the modular OpenRouter actions.
    function spliceBytes(bytes memory dst, uint256 dstOffset, bytes memory src, uint256 srcOffset, uint256 length)
        internal
        pure
    {
        if (length == 0) {
            revert SpliceLengthZero();
        }
        // unchecked: revert paths use checked add to keep things readable.
        if (srcOffset + length > src.length) {
            revert SpliceSrcOutOfBounds();
        }
        if (dstOffset + length > dst.length) {
            revert SpliceDstOutOfBounds();
        }
        assembly ("memory-safe") {
            mcopy(add(add(dst, 0x20), dstOffset), add(add(src, 0x20), srcOffset), length)
        }
    }
}


// ===== FILE: src/common/lib/CurrencyLib.sol =====
// SPDX-License-Identifier: GPL-3.0-only
pragma solidity 0.8.34;

import {SafeTransferLib} from "solady/src/utils/SafeTransferLib.sol";

error TransferFailed();

// @audit Audited before by Hexens: https://github.com/SocketDotTech/audits/blob/main/Bungee/12-2024%20-%20Bungee%20Protocol%20-%20Hexens.pdf
/// @title CurrencyLib
/// @notice Token transfer + balance helpers that treat the canonical native
///         pseudo-token (`0xEee...EEe`) the same way as the marketplace's
///         CurrencyLib. Backed by Solady SafeTransferLib for ERC20.
library CurrencyLib {
    /// @dev address used to identify native token
    address internal constant NATIVE_TOKEN_ADDRESS = address(0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE);

    function balanceOf(address token, address addr) internal view returns (uint256 balance) {
        if (token == NATIVE_TOKEN_ADDRESS) {
            balance = addr.balance;
        } else {
            balance = SafeTransferLib.balanceOf(token, addr);
        }
    }

    function transferFrom(address token, address from, address recipient, uint256 amount) internal {
        if (token == NATIVE_TOKEN_ADDRESS) {
            _transferNative(recipient, amount);
        } else {
            SafeTransferLib.safeTransferFrom(token, from, recipient, amount);
        }
    }

    function transfer(address token, address recipient, uint256 amount) internal {
        if (token == NATIVE_TOKEN_ADDRESS) {
            _transferNative(recipient, amount);
        } else {
            SafeTransferLib.safeTransfer(token, recipient, amount);
        }
    }

    function _transferNative(address recipient, uint256 amount) private {
        (bool success,) = recipient.call{value: amount, gas: 27_000}("");
        if (!success) {
            revert TransferFailed();
        }
    }
}


// ===== FILE: src/common/lib/RescueFundsLib.sol =====
// SPDX-License-Identifier: GPL-3.0-only
pragma solidity 0.8.34;

// @audit Audited before by Zellic: https://github.com/SocketDotTech/audits/blob/main/Socket-DL/07-2023%20-%20Data%20Layer%20-%20Zellic.pdf
import {SafeTransferLib} from "solady/src/utils/SafeTransferLib.sol";

error ZeroAddress();

/// @title RescueFundsLib
/// @notice Pull tokens or native ETH from the calling contract to a recipient.
library RescueFundsLib {
    address public constant ETH_ADDRESS = address(0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE);

    error InvalidTokenAddress();

    /// @param token_ ERC20 token or `ETH_ADDRESS` for native balance.
    /// @param rescueTo_ Recipient; must not be zero.
    /// @param amount_ Amount to transfer out of `address(this)`.
    function rescueFunds(address token_, address rescueTo_, uint256 amount_) internal {
        if (rescueTo_ == address(0)) {
            revert ZeroAddress();
        }

        if (token_ == ETH_ADDRESS) {
            SafeTransferLib.safeTransferETH(rescueTo_, amount_);
        } else {
            if (token_.code.length == 0) {
                revert InvalidTokenAddress();
            }
            SafeTransferLib.safeTransfer(token_, rescueTo_, amount_);
        }
    }
}


// ===== FILE: src/common/AccessRoles.sol =====
// SPDX-License-Identifier: GPL-3.0-only
pragma solidity 0.8.34;

bytes32 constant RESCUE_ROLE = keccak256("RESCUE_ROLE");


// ===== FILE: src/common/utils/Ownable.sol =====
// SPDX-License-Identifier: GPL-3.0-only
pragma solidity 0.8.34;

// @audit Audited before by Zellic: https://github.com/SocketDotTech/audits/blob/main/Socket-DL/07-2023%20-%20Data%20Layer%20-%20Zellic.pdf
/// @title Ownable
/// @notice Two-step ownership transfer, ported from
///         marketplace/src/utils/Ownable.sol. Simpler than OpenZeppelin's
///         `Ownable2Step` and matches the rest of the Bungee codebase.
abstract contract Ownable {
    error OnlyOwner();
    error OnlyNominee();

    address private _owner;
    address private _nominee;

    event OwnerNominated(address indexed nominee);
    event OwnerClaimed(address indexed claimer);

    constructor(address owner_) {
        _claimOwner(owner_);
    }

    modifier onlyOwner() {
        if (msg.sender != _owner) {
            revert OnlyOwner();
        }
        _;
    }

    function owner() public view returns (address) {
        return _owner;
    }

    function nominee() public view returns (address) {
        return _nominee;
    }

    function nominateOwner(address nominee_) external {
        if (msg.sender != _owner) {
            revert OnlyOwner();
        }
        _nominee = nominee_;
        emit OwnerNominated(nominee_);
    }

    function claimOwner() external {
        if (msg.sender != _nominee) {
            revert OnlyNominee();
        }
        _claimOwner(msg.sender);
    }

    function _claimOwner(address claimer_) internal {
        _owner = claimer_;
        _nominee = address(0);
        emit OwnerClaimed(claimer_);
    }
}
