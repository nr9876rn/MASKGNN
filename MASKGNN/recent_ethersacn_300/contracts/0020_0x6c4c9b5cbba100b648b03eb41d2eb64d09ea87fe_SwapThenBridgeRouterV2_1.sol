// ===== FILE: src/SwapThenBridgeRouterV2_1.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/// @title SwapThenBridgeRouterV2_1 — bridge-agnostic preferred-chain delivery, MultiHopRouter-aware
/// @notice Functionally identical to v2.0. The only behavioral difference is at
///         deployment: the constructor's `_routers` list MUST include the
///         relay's `MultiHopRouter` contract address alongside the underlying
///         DEX routers (PancakeSwap V2, Uniswap V3 SwapRouter02, Camelot
///         Algebra, etc).
///
///         This unblocks cross-protocol multi-hop atomic execution (e.g.
///         CAKE→WETH via V2 then WETH→USDC via V3 in ONE swap call) without
///         requiring multiple user signatures.
///
///         v2.0 silently failed for these routes with `InvalidRouter` because
///         `MultiHopRouter` was not in the whitelist set at v2.0 deployment
///         time. The relay started wrapping single-hop calls in
///         `MultiHopRouter.executeMultiHop(...)` for code uniformity, hitting
///         the missing-whitelist case for all routes — that's the bug this
///         redeploy fixes at the contract layer.
///
///         The relay's leg-rewrite logic (added in voidpool-invalid-router-fix)
///         continues to work as a belt-and-suspenders safety net — it unwraps
///         single-hop MultiHopRouter calls into direct DEX calls. With this
///         v2.1 deployment, the rewrite is no longer required for correctness
///         but stays in code as defense-in-depth (multiple paths to a working
///         swap is better than one).
///
///         IMPORTANT DEPLOYMENT NOTE:
///         If you deploy this contract without `MultiHopRouter` in `_routers`,
///         you've reproduced the exact v2.0 bug. The deployment script in
///         `script/Deploy.s.sol` enforces inclusion; do not bypass it.
///
/// @dev    NO owner. NO admin. IMMUTABLE bridge/router whitelist set at construction.
///         Same exact execution semantics as v2.0 — only the constructor inputs
///         differ at deployment time.
///
/// @author VoidPool Team
contract SwapThenBridgeRouterV2_1 is ReentrancyGuard {
    using SafeERC20 for IERC20;

    /// @notice On-chain version identifier. Lets the relay query the contract
    ///         and confirm it's talking to v2.1 (and therefore MultiHopRouter
    ///         is expected to be whitelisted) vs the v2.0 deployment.
    string public constant VERSION = "2.1.0";

    error ZeroAmount();
    error SwapFailed(address router);
    error BridgeFailed(address bridge);
    error BridgeOutputFailed(address bridge);
    error TooManyLegs();
    error ETHTransferFailed();
    error InvalidRouter(address router);
    error InvalidBridge(address bridge);
    error InsufficientOutputForBridge();

    event SwapThenBridgeExecutedV2(
        address indexed user,
        address inputToken,
        uint256 totalInput,
        uint256 sameChainLegs,
        uint256 crossChainLegs,
        address outputToken,
        uint256 outputBridged,
        address outputBridge,
        uint256 destChainId
    );

    // Per-router/bridge approval/whitelist - same pattern as SplitBridgeRouter.
    // Set ONCE at construction; no setter (immutable trust set).
    mapping(address => bool) public allowedRouters;
    mapping(address => bool) public allowedBridges;
    address public immutable WETH;

    struct SameChainLeg {
        address router;
        uint256 inputAmount;
        uint256 minOutput;
        bytes swapCalldata;
    }

    struct CrossChainLeg {
        address bridge;
        uint256 inputAmount;
        uint256 destChainId;
        uint256 nativeValue;
        bytes bridgeCalldata;
    }

    /// @notice Output-side bridge: how the contract delivers swap output to
    ///         the user's preferred chain. ANY bridge (Across, Stargate,
    ///         Wormhole, deBridge, Squid, Socket, LayerZero OFT) is valid
    ///         as long as it's in `allowedBridges`.
    struct BridgeOutputParams {
        address outputToken;        // Token to bridge (WETH, USDC, etc.)
        uint256 destChainId;        // Preferred chain for analytics (the bridge calldata
                                    // already encodes the dest; this field is for the event)
        address bridge;             // Bridge contract address (must be allowedBridges)
        uint256 nativeValue;        // ETH to forward as msg.value to the bridge (LZ fees etc.)
        uint256 minBridgeAmount;    // Floor on the amount we hand to the bridge — reverts
                                    // if the swap output is below this (user-controlled slippage)
        bytes   bridgeCalldata;     // Pre-built calldata for the bridge call. Relay encodes
                                    // either depositV3 (Across), sendToken (Stargate), or
                                    // transferTokens (Wormhole), etc., as appropriate.
        uint32  bridgeAmountOffset; // Byte offset in bridgeCalldata where the amount lives,
                                    // so the contract can patch the ACTUAL received amount
                                    // after the swap. Use 0 to skip patching.
    }

    /// @param _routers   Whitelisted DEX router addresses. MUST include the relay's
    ///                   `MultiHopRouter` address alongside the underlying DEX
    ///                   routers (PancakeSwap V2, Uniswap V3 SwapRouter02, Camelot
    ///                   Algebra, etc). Omitting MultiHopRouter reproduces the v2.0
    ///                   `InvalidRouter` bug — see the contract notice above.
    /// @param _bridges   Whitelisted bridge contracts (Across SpokePool, Stargate
    ///                   pool, Wormhole TokenBridge, deBridge gate, LayerZero OFT
    ///                   adapter, etc).
    /// @param _weth      Canonical WETH on this chain. Used by callers that wrap
    ///                   native ETH before depositing (the contract itself does
    ///                   not wrap; callers do so via SplitBridgeRouter's
    ///                   `executeSplitETH`).
    constructor(
        address[] memory _routers,
        address[] memory _bridges,
        address _weth
    ) {
        for (uint256 i; i < _routers.length; i++) {
            allowedRouters[_routers[i]] = true;
        }
        for (uint256 i; i < _bridges.length; i++) {
            allowedBridges[_bridges[i]] = true;
        }
        WETH = _weth;
    }

    /// @notice Execute split swap + bridge ALL swap output to preferred chain
    ///         via the bridge specified in `bridgeParams`.
    function execute(
        address inputToken,
        SameChainLeg[] calldata sameChainLegs,
        CrossChainLeg[] calldata crossChainLegs,
        BridgeOutputParams calldata bridgeParams
    ) external payable nonReentrant {
        uint256 totalSame = sameChainLegs.length;
        uint256 totalCross = crossChainLegs.length;
        if (totalSame + totalCross == 0) revert ZeroAmount();
        if (totalSame + totalCross > 10) revert TooManyLegs();

        // Calculate total input
        uint256 totalInput;
        for (uint256 i; i < totalSame; i++) totalInput += sameChainLegs[i].inputAmount;
        for (uint256 i; i < totalCross; i++) totalInput += crossChainLegs[i].inputAmount;
        if (totalInput == 0) revert ZeroAmount();

        // Pull input from user (single transfer for all legs)
        IERC20(inputToken).safeTransferFrom(msg.sender, address(this), totalInput);

        // ── Same-chain swap legs ──────────────────────────────
        // Output accumulates IN THIS CONTRACT (not sent to user yet)
        for (uint256 i; i < totalSame; i++) {
            SameChainLeg calldata leg = sameChainLegs[i];
            if (!allowedRouters[leg.router]) revert InvalidRouter(leg.router);

            IERC20(inputToken).forceApprove(leg.router, leg.inputAmount);

            uint256 outBefore = IERC20(bridgeParams.outputToken).balanceOf(address(this));
            (bool success,) = leg.router.call(leg.swapCalldata);
            if (!success) revert SwapFailed(leg.router);

            uint256 received = IERC20(bridgeParams.outputToken).balanceOf(address(this)) - outBefore;
            if (received < leg.minOutput) revert SwapFailed(leg.router);

            IERC20(inputToken).forceApprove(leg.router, 0);
        }

        // ── Bridge same-chain swap output to preferred chain ──
        // The bridge is now selected by the user (relay-built calldata)
        // instead of hardcoded to Across. Supports any whitelisted bridge.
        uint256 outputBridged = 0;
        if (totalSame > 0 && bridgeParams.destChainId != 0 && bridgeParams.bridge != address(0)) {
            if (!allowedBridges[bridgeParams.bridge]) revert InvalidBridge(bridgeParams.bridge);

            uint256 outputBalance = IERC20(bridgeParams.outputToken).balanceOf(address(this));
            if (outputBalance > 0) {
                // User-controlled floor on the amount entering the bridge.
                // This is what wires the frontend slippage % into the bridge step.
                if (outputBalance < bridgeParams.minBridgeAmount) revert InsufficientOutputForBridge();

                // Patch the bridge calldata with the ACTUAL received output amount
                // at the offset the relay specified. For Across depositV3, the amount
                // is the 5th arg (after depositor/recipient/inputToken/outputToken),
                // so offset = 4 + 32*4 = 132. For Stargate sendToken (SendParam tuple),
                // the offset is deeper because of tuple encoding — the relay computes it.
                if (bridgeParams.bridgeAmountOffset > 0) {
                    bytes memory bcd = bridgeParams.bridgeCalldata; // memory copy for patching
                    uint32 off = bridgeParams.bridgeAmountOffset;
                    assembly {
                        // bcd points to length-prefixed bytes; +32 to skip length
                        mstore(add(add(bcd, 32), off), outputBalance)
                    }
                    IERC20(bridgeParams.outputToken).forceApprove(bridgeParams.bridge, outputBalance);
                    (bool success,) = bridgeParams.bridge.call{value: bridgeParams.nativeValue}(bcd);
                    if (!success) revert BridgeOutputFailed(bridgeParams.bridge);
                } else {
                    IERC20(bridgeParams.outputToken).forceApprove(bridgeParams.bridge, outputBalance);
                    (bool success,) = bridgeParams.bridge.call{value: bridgeParams.nativeValue}(bridgeParams.bridgeCalldata);
                    if (!success) revert BridgeOutputFailed(bridgeParams.bridge);
                }

                IERC20(bridgeParams.outputToken).forceApprove(bridgeParams.bridge, 0);
                outputBridged = outputBalance;
            }
        }

        // ── Cross-chain legs (bridge INPUT to intermediate chains) ──
        // Same as SplitBridgeRouter — input token gets bridged with relay-built
        // calldata for any whitelisted bridge. SwapAndBridgeV4 on the dest chain
        // handles the dest swap + optional bridge-forward.
        for (uint256 i; i < totalCross; i++) {
            CrossChainLeg calldata leg = crossChainLegs[i];
            if (!allowedBridges[leg.bridge]) revert InvalidBridge(leg.bridge);
            IERC20(inputToken).forceApprove(leg.bridge, leg.inputAmount);

            (bool success,) = leg.bridge.call{value: leg.nativeValue}(leg.bridgeCalldata);
            if (!success) revert BridgeFailed(leg.bridge);

            IERC20(inputToken).forceApprove(leg.bridge, 0);
        }

        // Return any dust
        uint256 remaining = IERC20(inputToken).balanceOf(address(this));
        if (remaining > 0) IERC20(inputToken).safeTransfer(msg.sender, remaining);

        uint256 outputDust = IERC20(bridgeParams.outputToken).balanceOf(address(this));
        if (outputDust > 0) IERC20(bridgeParams.outputToken).safeTransfer(msg.sender, outputDust);

        uint256 ethLeft = address(this).balance;
        if (ethLeft > 0) {
            (bool sent,) = msg.sender.call{value: ethLeft}("");
            if (!sent) revert ETHTransferFailed();
        }

        emit SwapThenBridgeExecutedV2(
            msg.sender, inputToken, totalInput,
            totalSame, totalCross, bridgeParams.outputToken,
            outputBridged, bridgeParams.bridge, bridgeParams.destChainId
        );
    }

    /// @notice Rescue stuck tokens. Permissionless — anyone can trigger
    ///         transfer of stuck tokens to anyone. Stuck tokens are an
    ///         operational concern, not value at risk: the contract holds
    ///         tokens only transiently inside `execute()` (protected by
    ///         `nonReentrant`). Anything left between calls is leftover
    ///         dust from a partially-executed prior swap.
    function rescueTokens(address token, address recipient) external nonReentrant {
        uint256 bal = IERC20(token).balanceOf(address(this));
        if (bal == 0) revert ZeroAmount();
        IERC20(token).safeTransfer(recipient, bal);
    }

    function isRouterAllowed(address r) external view returns (bool) { return allowedRouters[r]; }
    function isBridgeAllowed(address b) external view returns (bool) { return allowedBridges[b]; }

    receive() external payable {}
}


// ===== FILE: lib/calibur/lib/openzeppelin-contracts/contracts/token/ERC20/IERC20.sol =====
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


// ===== FILE: lib/calibur/lib/openzeppelin-contracts/contracts/token/ERC20/utils/SafeERC20.sol =====
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


// ===== FILE: lib/calibur/lib/openzeppelin-contracts/contracts/utils/ReentrancyGuard.sol =====
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


// ===== FILE: lib/calibur/lib/openzeppelin-contracts/contracts/interfaces/IERC1363.sol =====
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


// ===== FILE: lib/calibur/lib/openzeppelin-contracts/contracts/interfaces/IERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (interfaces/IERC20.sol)

pragma solidity ^0.8.20;

import {IERC20} from "../token/ERC20/IERC20.sol";


// ===== FILE: lib/calibur/lib/openzeppelin-contracts/contracts/interfaces/IERC165.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (interfaces/IERC165.sol)

pragma solidity ^0.8.20;

import {IERC165} from "../utils/introspection/IERC165.sol";


// ===== FILE: lib/calibur/lib/openzeppelin-contracts/contracts/utils/introspection/IERC165.sol =====
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
