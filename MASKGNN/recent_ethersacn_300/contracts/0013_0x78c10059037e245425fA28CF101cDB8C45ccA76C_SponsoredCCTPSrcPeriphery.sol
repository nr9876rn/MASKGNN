// ===== FILE: contracts/periphery/mintburn/sponsored-cctp/SponsoredCCTPSrcPeriphery.sol =====
//SPDX-License-Identifier: Unlicense
pragma solidity ^0.8.0;

import { Ownable } from "@openzeppelin/contracts-v4/access/Ownable.sol";
import { IERC20 } from "@openzeppelin/contracts-v4/token/ERC20/IERC20.sol";
import { SafeERC20 } from "@openzeppelin/contracts-v4/token/ERC20/utils/SafeERC20.sol";
import { Address } from "@openzeppelin/contracts-v4/utils/Address.sol";
import { ITokenMessengerV2 } from "../../../external/interfaces/CCTPInterfaces.sol";

import { SponsoredCCTPQuoteLib } from "../../../libraries/SponsoredCCTPQuoteLib.sol";
import { SponsoredCCTPInterface } from "../../../interfaces/SponsoredCCTPInterface.sol";
import { SponsoredCCTPDstPeriphery } from "./SponsoredCCTPDstPeriphery.sol";
import { Bytes32ToAddress } from "../../../libraries/AddressConverters.sol";

/**
 * @title SponsoredCCTPSrcPeriphery
 * @notice Source chain periphery contract that supports sponsored/non-sponsored CCTP deposits.
 * @dev This contract is used to deposit tokens for burn via CCTP.
 */
contract SponsoredCCTPSrcPeriphery is SponsoredCCTPInterface, Ownable {
    using SafeERC20 for IERC20;
    using Bytes32ToAddress for bytes32;
    using Address for address;

    /// @notice The CCTP token messenger contract.
    ITokenMessengerV2 public immutable cctpTokenMessenger;

    /// @notice The source domain ID for the chain that this contract is deployed on.
    uint32 public immutable sourceDomain;

    /// @custom:storage-location erc7201:SponsoredCCTPSrcPeriphery.main
    struct MainStorage {
        /// @notice The signer address that is used to validate the signatures of the quotes.
        address signer;
        /// @notice A mapping of used nonces to prevent replay attacks.
        mapping(bytes32 => bool) usedNonces;
    }

    // keccak256(abi.encode(uint256(keccak256("erc7201:SponsoredCCTPSrcPeriphery.main")) - 1)) & ~bytes32(uint256(0xff));
    bytes32 private constant MAIN_STORAGE_LOCATION = 0xf0a1b42b86a218bb35dbc2254545839ce4b1bf1d3780b5099e3e0abfc7a5b200;

    function _getMainStorage() private pure returns (MainStorage storage $) {
        assembly {
            $.slot := MAIN_STORAGE_LOCATION
        }
    }

    /**
     * @notice Constructor for the SponsoredCCTPSrcPeriphery contract.
     * @param _cctpTokenMessenger The address of the CCTP token messenger contract.
     * @param _sourceDomain The source domain ID for the chain that this contract is deployed on.
     * @param _signer The signer address that is used to validate the signatures of the quotes.
     */
    constructor(address _cctpTokenMessenger, uint32 _sourceDomain, address _signer) {
        cctpTokenMessenger = ITokenMessengerV2(_cctpTokenMessenger);
        sourceDomain = _sourceDomain;
        _getMainStorage().signer = _signer;
    }

    /**
     * @notice Returns the signer address that is used to validate the signatures of the quotes.
     * @return The signer address.
     */
    function signer() external view returns (address) {
        return _getMainStorage().signer;
    }

    /**
     * @notice Returns true if the nonce has been used, false otherwise.
     * @param nonce The nonce to check.
     * @return True if the nonce has been used, false otherwise.
     */
    function usedNonces(bytes32 nonce) external view returns (bool) {
        return _getMainStorage().usedNonces[nonce];
    }

    /**
     * @notice Deposits tokens for burn via CCTP.
     * @param quote The quote that contains the data for the deposit.
     * @param signature The signature of the quote.
     */
    function depositForBurn(SponsoredCCTPInterface.SponsoredCCTPQuote memory quote, bytes memory signature) external {
        MainStorage storage $ = _getMainStorage();
        if (!SponsoredCCTPQuoteLib.validateSignature($.signer, quote, signature)) revert InvalidSignature();
        if ($.usedNonces[quote.nonce]) revert InvalidNonce();
        if (quote.deadline < block.timestamp) revert InvalidDeadline();
        if (quote.sourceDomain != sourceDomain) revert InvalidSourceDomain();

        $.usedNonces[quote.nonce] = true;

        if (quote.destinationDomain == sourceDomain) {
            address destinationHandler = quote.mintRecipient.toAddress();
            if (!destinationHandler.isContract()) revert InvalidDirectHandler();

            address burnToken = quote.burnToken.toAddress();
            IERC20(burnToken).safeTransferFrom(msg.sender, destinationHandler, quote.amount);
            SponsoredCCTPDstPeriphery(payable(destinationHandler)).directReceiveMessage(quote);

            emit SponsoredCCTPDirectExecution(
                quote.nonce,
                msg.sender,
                quote.finalRecipient,
                quote.deadline,
                quote.maxBpsToSponsor,
                quote.maxUserSlippageBps,
                quote.finalToken,
                quote.destinationDex,
                quote.accountCreationMode,
                signature
            );
        } else {
            (
                uint256 amount,
                uint32 destinationDomain,
                bytes32 mintRecipient,
                address burnToken,
                bytes32 destinationCaller,
                uint256 maxFee,
                uint32 minFinalityThreshold,
                bytes memory hookData
            ) = SponsoredCCTPQuoteLib.getDepositForBurnData(quote);

            IERC20(burnToken).safeTransferFrom(msg.sender, address(this), amount);
            IERC20(burnToken).forceApprove(address(cctpTokenMessenger), amount);

            cctpTokenMessenger.depositForBurnWithHook(
                amount,
                destinationDomain,
                mintRecipient,
                burnToken,
                destinationCaller,
                maxFee,
                minFinalityThreshold,
                hookData
            );

            emit SponsoredDepositForBurn(
                quote.nonce,
                msg.sender,
                quote.finalRecipient,
                quote.deadline,
                quote.maxBpsToSponsor,
                quote.maxUserSlippageBps,
                quote.finalToken,
                quote.destinationDex,
                quote.accountCreationMode,
                signature
            );
        }
    }

    /**
     * @notice Sets the signer address that is used to validate the signatures of the quotes.
     * @param _signer The new signer address.
     */
    function setSigner(address _signer) external onlyOwner {
        _getMainStorage().signer = _signer;
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts-v4/access/Ownable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.0) (access/Ownable.sol)

pragma solidity ^0.8.0;

import "../utils/Context.sol";

/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * By default, the owner account will be the one that deploys the contract. This
 * can later be changed with {transferOwnership}.
 *
 * This module is used through inheritance. It will make available the modifier
 * `onlyOwner`, which can be applied to your functions to restrict their use to
 * the owner.
 */
abstract contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev Initializes the contract setting the deployer as the initial owner.
     */
    constructor() {
        _transferOwnership(_msgSender());
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
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
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
        require(newOwner != address(0), "Ownable: new owner is the zero address");
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


// ===== FILE: node_modules/_openzeppelin/contracts-v4/token/ERC20/IERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.0) (token/ERC20/IERC20.sol)

pragma solidity ^0.8.0;

/**
 * @dev Interface of the ERC20 standard as defined in the EIP.
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
     * @dev Returns the amount of tokens in existence.
     */
    function totalSupply() external view returns (uint256);

    /**
     * @dev Returns the amount of tokens owned by `account`.
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Moves `amount` tokens from the caller's account to `to`.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transfer(address to, uint256 amount) external returns (bool);

    /**
     * @dev Returns the remaining number of tokens that `spender` will be
     * allowed to spend on behalf of `owner` through {transferFrom}. This is
     * zero by default.
     *
     * This value changes when {approve} or {transferFrom} are called.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Sets `amount` as the allowance of `spender` over the caller's tokens.
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
    function approve(address spender, uint256 amount) external returns (bool);

    /**
     * @dev Moves `amount` tokens from `from` to `to` using the
     * allowance mechanism. `amount` is then deducted from the caller's
     * allowance.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}


// ===== FILE: node_modules/_openzeppelin/contracts-v4/token/ERC20/utils/SafeERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.3) (token/ERC20/utils/SafeERC20.sol)

pragma solidity ^0.8.0;

import "../IERC20.sol";
import "../extensions/IERC20Permit.sol";
import "../../../utils/Address.sol";

/**
 * @title SafeERC20
 * @dev Wrappers around ERC20 operations that throw on failure (when the token
 * contract returns false). Tokens that return no value (and instead revert or
 * throw on failure) are also supported, non-reverting calls are assumed to be
 * successful.
 * To use this library you can add a `using SafeERC20 for IERC20;` statement to your contract,
 * which allows you to call the safe operations as `token.safeTransfer(...)`, etc.
 */
library SafeERC20 {
    using Address for address;

    /**
     * @dev Transfer `value` amount of `token` from the calling contract to `to`. If `token` returns no value,
     * non-reverting calls are assumed to be successful.
     */
    function safeTransfer(IERC20 token, address to, uint256 value) internal {
        _callOptionalReturn(token, abi.encodeWithSelector(token.transfer.selector, to, value));
    }

    /**
     * @dev Transfer `value` amount of `token` from `from` to `to`, spending the approval given by `from` to the
     * calling contract. If `token` returns no value, non-reverting calls are assumed to be successful.
     */
    function safeTransferFrom(IERC20 token, address from, address to, uint256 value) internal {
        _callOptionalReturn(token, abi.encodeWithSelector(token.transferFrom.selector, from, to, value));
    }

    /**
     * @dev Deprecated. This function has issues similar to the ones found in
     * {IERC20-approve}, and its usage is discouraged.
     *
     * Whenever possible, use {safeIncreaseAllowance} and
     * {safeDecreaseAllowance} instead.
     */
    function safeApprove(IERC20 token, address spender, uint256 value) internal {
        // safeApprove should only be called when setting an initial allowance,
        // or when resetting it to zero. To increase and decrease it, use
        // 'safeIncreaseAllowance' and 'safeDecreaseAllowance'
        require(
            (value == 0) || (token.allowance(address(this), spender) == 0),
            "SafeERC20: approve from non-zero to non-zero allowance"
        );
        _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, value));
    }

    /**
     * @dev Increase the calling contract's allowance toward `spender` by `value`. If `token` returns no value,
     * non-reverting calls are assumed to be successful.
     */
    function safeIncreaseAllowance(IERC20 token, address spender, uint256 value) internal {
        uint256 oldAllowance = token.allowance(address(this), spender);
        _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, oldAllowance + value));
    }

    /**
     * @dev Decrease the calling contract's allowance toward `spender` by `value`. If `token` returns no value,
     * non-reverting calls are assumed to be successful.
     */
    function safeDecreaseAllowance(IERC20 token, address spender, uint256 value) internal {
        unchecked {
            uint256 oldAllowance = token.allowance(address(this), spender);
            require(oldAllowance >= value, "SafeERC20: decreased allowance below zero");
            _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, oldAllowance - value));
        }
    }

    /**
     * @dev Set the calling contract's allowance toward `spender` to `value`. If `token` returns no value,
     * non-reverting calls are assumed to be successful. Meant to be used with tokens that require the approval
     * to be set to zero before setting it to a non-zero value, such as USDT.
     */
    function forceApprove(IERC20 token, address spender, uint256 value) internal {
        bytes memory approvalCall = abi.encodeWithSelector(token.approve.selector, spender, value);

        if (!_callOptionalReturnBool(token, approvalCall)) {
            _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, 0));
            _callOptionalReturn(token, approvalCall);
        }
    }

    /**
     * @dev Use a ERC-2612 signature to set the `owner` approval toward `spender` on `token`.
     * Revert on invalid signature.
     */
    function safePermit(
        IERC20Permit token,
        address owner,
        address spender,
        uint256 value,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) internal {
        uint256 nonceBefore = token.nonces(owner);
        token.permit(owner, spender, value, deadline, v, r, s);
        uint256 nonceAfter = token.nonces(owner);
        require(nonceAfter == nonceBefore + 1, "SafeERC20: permit did not succeed");
    }

    /**
     * @dev Imitates a Solidity high-level call (i.e. a regular function call to a contract), relaxing the requirement
     * on the return value: the return value is optional (but if data is returned, it must not be false).
     * @param token The token targeted by the call.
     * @param data The call data (encoded using abi.encode or one of its variants).
     */
    function _callOptionalReturn(IERC20 token, bytes memory data) private {
        // We need to perform a low level call here, to bypass Solidity's return data size checking mechanism, since
        // we're implementing it ourselves. We use {Address-functionCall} to perform this call, which verifies that
        // the target address contains contract code and also asserts for success in the low-level call.

        bytes memory returndata = address(token).functionCall(data, "SafeERC20: low-level call failed");
        require(returndata.length == 0 || abi.decode(returndata, (bool)), "SafeERC20: ERC20 operation did not succeed");
    }

    /**
     * @dev Imitates a Solidity high-level call (i.e. a regular function call to a contract), relaxing the requirement
     * on the return value: the return value is optional (but if data is returned, it must not be false).
     * @param token The token targeted by the call.
     * @param data The call data (encoded using abi.encode or one of its variants).
     *
     * This is a variant of {_callOptionalReturn} that silents catches all reverts and returns a bool instead.
     */
    function _callOptionalReturnBool(IERC20 token, bytes memory data) private returns (bool) {
        // We need to perform a low level call here, to bypass Solidity's return data size checking mechanism, since
        // we're implementing it ourselves. We cannot use {Address-functionCall} here since this should return false
        // and not revert is the subcall reverts.

        (bool success, bytes memory returndata) = address(token).call(data);
        return
            success && (returndata.length == 0 || abi.decode(returndata, (bool))) && Address.isContract(address(token));
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts-v4/utils/Address.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.0) (utils/Address.sol)

pragma solidity ^0.8.1;

/**
 * @dev Collection of functions related to the address type
 */
library Address {
    /**
     * @dev Returns true if `account` is a contract.
     *
     * [IMPORTANT]
     * ====
     * It is unsafe to assume that an address for which this function returns
     * false is an externally-owned account (EOA) and not a contract.
     *
     * Among others, `isContract` will return false for the following
     * types of addresses:
     *
     *  - an externally-owned account
     *  - a contract in construction
     *  - an address where a contract will be created
     *  - an address where a contract lived, but was destroyed
     *
     * Furthermore, `isContract` will also return true if the target contract within
     * the same transaction is already scheduled for destruction by `SELFDESTRUCT`,
     * which only has an effect at the end of a transaction.
     * ====
     *
     * [IMPORTANT]
     * ====
     * You shouldn't rely on `isContract` to protect against flash loan attacks!
     *
     * Preventing calls from contracts is highly discouraged. It breaks composability, breaks support for smart wallets
     * like Gnosis Safe, and does not provide security since it can be circumvented by calling from a contract
     * constructor.
     * ====
     */
    function isContract(address account) internal view returns (bool) {
        // This method relies on extcodesize/address.code.length, which returns 0
        // for contracts in construction, since the code is only stored at the end
        // of the constructor execution.

        return account.code.length > 0;
    }

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
     * https://solidity.readthedocs.io/en/v0.8.0/security-considerations.html#use-the-checks-effects-interactions-pattern[checks-effects-interactions pattern].
     */
    function sendValue(address payable recipient, uint256 amount) internal {
        require(address(this).balance >= amount, "Address: insufficient balance");

        (bool success, ) = recipient.call{value: amount}("");
        require(success, "Address: unable to send value, recipient may have reverted");
    }

    /**
     * @dev Performs a Solidity function call using a low level `call`. A
     * plain `call` is an unsafe replacement for a function call: use this
     * function instead.
     *
     * If `target` reverts with a revert reason, it is bubbled up by this
     * function (like regular Solidity function calls).
     *
     * Returns the raw returned data. To convert to the expected return value,
     * use https://solidity.readthedocs.io/en/latest/units-and-global-variables.html?highlight=abi.decode#abi-encoding-and-decoding-functions[`abi.decode`].
     *
     * Requirements:
     *
     * - `target` must be a contract.
     * - calling `target` with `data` must not revert.
     *
     * _Available since v3.1._
     */
    function functionCall(address target, bytes memory data) internal returns (bytes memory) {
        return functionCallWithValue(target, data, 0, "Address: low-level call failed");
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`], but with
     * `errorMessage` as a fallback revert reason when `target` reverts.
     *
     * _Available since v3.1._
     */
    function functionCall(
        address target,
        bytes memory data,
        string memory errorMessage
    ) internal returns (bytes memory) {
        return functionCallWithValue(target, data, 0, errorMessage);
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but also transferring `value` wei to `target`.
     *
     * Requirements:
     *
     * - the calling contract must have an ETH balance of at least `value`.
     * - the called Solidity function must be `payable`.
     *
     * _Available since v3.1._
     */
    function functionCallWithValue(address target, bytes memory data, uint256 value) internal returns (bytes memory) {
        return functionCallWithValue(target, data, value, "Address: low-level call with value failed");
    }

    /**
     * @dev Same as {xref-Address-functionCallWithValue-address-bytes-uint256-}[`functionCallWithValue`], but
     * with `errorMessage` as a fallback revert reason when `target` reverts.
     *
     * _Available since v3.1._
     */
    function functionCallWithValue(
        address target,
        bytes memory data,
        uint256 value,
        string memory errorMessage
    ) internal returns (bytes memory) {
        require(address(this).balance >= value, "Address: insufficient balance for call");
        (bool success, bytes memory returndata) = target.call{value: value}(data);
        return verifyCallResultFromTarget(target, success, returndata, errorMessage);
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but performing a static call.
     *
     * _Available since v3.3._
     */
    function functionStaticCall(address target, bytes memory data) internal view returns (bytes memory) {
        return functionStaticCall(target, data, "Address: low-level static call failed");
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-string-}[`functionCall`],
     * but performing a static call.
     *
     * _Available since v3.3._
     */
    function functionStaticCall(
        address target,
        bytes memory data,
        string memory errorMessage
    ) internal view returns (bytes memory) {
        (bool success, bytes memory returndata) = target.staticcall(data);
        return verifyCallResultFromTarget(target, success, returndata, errorMessage);
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but performing a delegate call.
     *
     * _Available since v3.4._
     */
    function functionDelegateCall(address target, bytes memory data) internal returns (bytes memory) {
        return functionDelegateCall(target, data, "Address: low-level delegate call failed");
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-string-}[`functionCall`],
     * but performing a delegate call.
     *
     * _Available since v3.4._
     */
    function functionDelegateCall(
        address target,
        bytes memory data,
        string memory errorMessage
    ) internal returns (bytes memory) {
        (bool success, bytes memory returndata) = target.delegatecall(data);
        return verifyCallResultFromTarget(target, success, returndata, errorMessage);
    }

    /**
     * @dev Tool to verify that a low level call to smart-contract was successful, and revert (either by bubbling
     * the revert reason or using the provided one) in case of unsuccessful call or if target was not a contract.
     *
     * _Available since v4.8._
     */
    function verifyCallResultFromTarget(
        address target,
        bool success,
        bytes memory returndata,
        string memory errorMessage
    ) internal view returns (bytes memory) {
        if (success) {
            if (returndata.length == 0) {
                // only check isContract if the call was successful and the return data is empty
                // otherwise we already know that it was a contract
                require(isContract(target), "Address: call to non-contract");
            }
            return returndata;
        } else {
            _revert(returndata, errorMessage);
        }
    }

    /**
     * @dev Tool to verify that a low level call was successful, and revert if it wasn't, either by bubbling the
     * revert reason or using the provided one.
     *
     * _Available since v4.3._
     */
    function verifyCallResult(
        bool success,
        bytes memory returndata,
        string memory errorMessage
    ) internal pure returns (bytes memory) {
        if (success) {
            return returndata;
        } else {
            _revert(returndata, errorMessage);
        }
    }

    function _revert(bytes memory returndata, string memory errorMessage) private pure {
        // Look for revert reason and bubble it up if present
        if (returndata.length > 0) {
            // The easiest way to bubble the revert reason is using memory via assembly
            /// @solidity memory-safe-assembly
            assembly {
                let returndata_size := mload(returndata)
                revert(add(32, returndata), returndata_size)
            }
        } else {
            revert(errorMessage);
        }
    }
}


// ===== FILE: contracts/external/interfaces/CCTPInterfaces.sol =====
/**
 * Copyright (C) 2015, 2016, 2017 Dapphub
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8.0;

/**
 * Imported as-is from commit 139d8d0ce3b5531d3c7ec284f89d946dfb720016 of:
 *   * https://github.com/walkerq/evm-cctp-contracts/blob/139d8d0ce3b5531d3c7ec284f89d946dfb720016/src/TokenMessenger.sol
 * Changes applied post-import:
 *   * Removed a majority of code from this contract and converted the needed function signatures in this interface.
 */
interface ITokenMessenger {
    /**
     * @notice Deposits and burns tokens from sender to be minted on destination domain.
     * Emits a `DepositForBurn` event.
     * @dev reverts if:
     * - given burnToken is not supported
     * - given destinationDomain has no TokenMessenger registered
     * - transferFrom() reverts. For example, if sender's burnToken balance or approved allowance
     * to this contract is less than `amount`.
     * - burn() reverts. For example, if `amount` is 0.
     * - MessageTransmitter returns false or reverts.
     * @param amount amount of tokens to burn
     * @param destinationDomain destination domain
     * @param mintRecipient address of mint recipient on destination domain
     * @param burnToken address of contract to burn deposited tokens, on local domain
     * @return _nonce unique nonce reserved by message
     */
    function depositForBurn(
        uint256 amount,
        uint32 destinationDomain,
        bytes32 mintRecipient,
        address burnToken
    ) external returns (uint64 _nonce);

    /**
     * @notice Minter responsible for minting and burning tokens on the local domain
     * @dev A TokenMessenger stores a TokenMinter contract which extends the TokenController contract.
     * https://github.com/circlefin/evm-cctp-contracts/blob/817397db0a12963accc08ff86065491577bbc0e5/src/TokenMessenger.sol#L110
     * @return minter Token Minter contract.
     */
    function localMinter() external view returns (ITokenMinter minter);
}

interface ITokenMessengerV2 {
    /**
     * @notice Minter responsible for minting and burning tokens on the local domain.
     * Used on destination to prove that a received message would mint via the expected
     * TokenMinter, and to resolve the local token that corresponds to a remote burnToken.
     * https://github.com/circlefin/evm-cctp-contracts/blob/63ab1f0ac06ce0793c0bbfbb8d09816bc211386d/src/v2/TokenMessengerV2.sol
     */
    function localMinter() external view returns (ITokenMinter minter);

    // Source: https://github.com/circlefin/evm-cctp-contracts/blob/63ab1f0ac06ce0793c0bbfbb8d09816bc211386d/src/v2/TokenMessengerV2.sol#L138C1-L166C15
    /**
     * @notice Deposits and burns tokens from sender to be minted on destination domain.
     * Emits a `DepositForBurn` event.
     * @dev reverts if:
     * - given burnToken is not supported
     * - given destinationDomain has no TokenMessenger registered
     * - transferFrom() reverts. For example, if sender's burnToken balance or approved allowance
     * to this contract is less than `amount`.
     * - burn() reverts. For example, if `amount` is 0.
     * - maxFee is greater than or equal to `amount`.
     * - MessageTransmitterV2#sendMessage reverts.
     * @param amount amount of tokens to burn
     * @param destinationDomain destination domain to receive message on
     * @param mintRecipient address of mint recipient on destination domain
     * @param burnToken token to burn `amount` of, on local domain
     * @param destinationCaller authorized caller on the destination domain, as bytes32. If equal to bytes32(0),
     * any address can broadcast the message.
     * @param maxFee maximum fee to pay on the destination domain, specified in units of burnToken
     * @param minFinalityThreshold the minimum finality at which a burn message will be attested to.
     */
    function depositForBurn(
        uint256 amount,
        uint32 destinationDomain,
        bytes32 mintRecipient,
        address burnToken,
        bytes32 destinationCaller,
        uint256 maxFee,
        uint32 minFinalityThreshold
    ) external;

    // Source: https://github.com/circlefin/evm-cctp-contracts/blob/63ab1f0ac06ce0793c0bbfbb8d09816bc211386d/src/v2/TokenMessengerV2.sol#L180C1-L210C15
    /**
     * @notice Deposits and burns tokens from sender to be minted on destination domain.
     * Emits a `DepositForBurn` event.
     * @dev reverts if:
     * - `hookData` is zero-length
     * - `burnToken` is not supported
     * - `destinationDomain` has no TokenMessenger registered
     * - transferFrom() reverts. For example, if sender's burnToken balance or approved allowance
     * to this contract is less than `amount`.
     * - burn() reverts. For example, if `amount` is 0.
     * - maxFee is greater than or equal to `amount`.
     * - MessageTransmitterV2#sendMessage reverts.
     * @param amount amount of tokens to burn
     * @param destinationDomain destination domain to receive message on
     * @param mintRecipient address of mint recipient on destination domain, as bytes32
     * @param burnToken token to burn `amount` of, on local domain
     * @param destinationCaller authorized caller on the destination domain, as bytes32. If equal to bytes32(0),
     * any address can broadcast the message.
     * @param maxFee maximum fee to pay on the destination domain, specified in units of burnToken
     * @param hookData hook data to append to burn message for interpretation on destination domain
     */
    function depositForBurnWithHook(
        uint256 amount,
        uint32 destinationDomain,
        bytes32 mintRecipient,
        address burnToken,
        bytes32 destinationCaller,
        uint256 maxFee,
        uint32 minFinalityThreshold,
        bytes calldata hookData
    ) external;
}

/**
 * A TokenMessenger stores a TokenMinter contract which extends the TokenController contract. The TokenController
 * contract has a burnLimitsPerMessage public mapping which can be queried to find the per-message burn limit
 * for a given token:
 * https://github.com/circlefin/evm-cctp-contracts/blob/817397db0a12963accc08ff86065491577bbc0e5/src/TokenMinter.sol#L33
 * https://github.com/circlefin/evm-cctp-contracts/blob/817397db0a12963accc08ff86065491577bbc0e5/src/roles/TokenController.sol#L69C40-L69C60
 *
 */
interface ITokenMinter {
    /**
     * @notice Supported burnable tokens on the local domain
     * local token (address) => maximum burn amounts per message
     * @param token address of token contract
     * @return burnLimit maximum burn amount per message for token
     */
    function burnLimitsPerMessage(address token) external view returns (uint256);

    /**
     * @notice Returns the local token that is associated with the given remote burnToken on `remoteDomain`.
     * Returns address(0) when no link has been configured.
     * https://github.com/circlefin/evm-cctp-contracts/blob/63ab1f0ac06ce0793c0bbfbb8d09816bc211386d/src/TokenMinter.sol#L155
     * @param remoteDomain CCTP source domain
     * @param remoteToken burnToken (as bytes32) on the remote domain
     * @return localToken the locally-minted token, or address(0) if the pair is not linked
     */
    function getLocalToken(uint32 remoteDomain, bytes32 remoteToken) external view returns (address localToken);
}

/**
 * IMessageTransmitter in CCTP inherits IRelayer and IReceiver, but here we only import sendMessage from IRelayer:
 * https://github.com/circlefin/evm-cctp-contracts/blob/377c9bd813fb86a42d900ae4003599d82aef635a/src/interfaces/IMessageTransmitter.sol#L25
 * https://github.com/circlefin/evm-cctp-contracts/blob/377c9bd813fb86a42d900ae4003599d82aef635a/src/interfaces/IRelayer.sol#L23-L35
 */
interface IMessageTransmitter {
    /**
     * @notice Sends an outgoing message from the source domain.
     * @dev Increment nonce, format the message, and emit `MessageSent` event with message information.
     * @param destinationDomain Domain of destination chain
     * @param recipient Address of message recipient on destination domain as bytes32
     * @param messageBody Raw bytes content of message
     * @return nonce reserved by message
     */
    function sendMessage(
        uint32 destinationDomain,
        bytes32 recipient,
        bytes calldata messageBody
    ) external returns (uint64);
}

interface IMessageTransmitterV2 {
    // Source: https://github.com/circlefin/evm-cctp-contracts/blob/63ab1f0ac06ce0793c0bbfbb8d09816bc211386d/src/v2/MessageTransmitterV2.sol#L176C1-L209C61
    /**
     * @notice Receive a message. Messages can only be broadcast once for a given nonce.
     * The message body of a valid message is passed to the specified recipient for further processing.
     *
     * @dev Attestation format:
     * A valid attestation is the concatenated 65-byte signature(s) of exactly
     * `thresholdSignature` signatures, in increasing order of attester address.
     * ***If the attester addresses recovered from signatures are not in
     * increasing order, signature verification will fail.***
     * If incorrect number of signatures or duplicate signatures are supplied,
     * signature verification will fail.
     *
     * Message Format:
     *
     * Field                        Bytes      Type       Index
     * version                      4          uint32     0
     * sourceDomain                 4          uint32     4
     * destinationDomain            4          uint32     8
     * nonce                        32         bytes32    12
     * sender                       32         bytes32    44
     * recipient                    32         bytes32    76
     * destinationCaller            32         bytes32    108
     * minFinalityThreshold         4          uint32     140
     * finalityThresholdExecuted    4          uint32     144
     * messageBody                  dynamic    bytes      148
     * @param message Message bytes
     * @param attestation Concatenated 65-byte signature(s) of `message`, in increasing order
     * of the attester address recovered from signatures.
     * @return success True, if successful; false, if not
     */
    function receiveMessage(bytes calldata message, bytes calldata attestation) external returns (bool success);
}


// ===== FILE: contracts/libraries/SponsoredCCTPQuoteLib.sol =====
// SPDX-License-Identifier: BUSL-1.1
pragma solidity ^0.8.0;

import { SignatureChecker } from "@openzeppelin/contracts-v4/utils/cryptography/SignatureChecker.sol";

import { SponsoredCCTPInterface } from "../interfaces/SponsoredCCTPInterface.sol";
import { BytesLib } from "../external/libraries/BytesLib.sol";
import { Bytes32ToAddress } from "./AddressConverters.sol";

/**
 * @title SponsoredCCTPQuoteLib
 * @notice Library that contains the functions to get the data from the quotes and validate the signatures.
 */
library SponsoredCCTPQuoteLib {
    using BytesLib for bytes;
    using Bytes32ToAddress for bytes32;

    /// @dev Indices of each field in message that we get from CCTP
    /// Source: https://github.com/circlefin/evm-cctp-contracts/blob/4061786a5726bc05f99fcdb53b0985599f0dbaf7/src/messages/v2/MessageV2.sol#L52-L61
    uint256 private constant VERSION_INDEX = 0;
    uint256 private constant SOURCE_DOMAIN_INDEX = 4;
    uint256 private constant DESTINATION_DOMAIN_INDEX = 8;
    uint256 private constant NONCE_INDEX = 12;
    uint256 private constant SENDER_INDEX = 44;
    uint256 private constant RECIPIENT_INDEX = 76;
    uint256 private constant DESTINATION_CALLER_INDEX = 108;
    uint256 private constant MIN_FINALITY_THRESHOLD_INDEX = 140;
    uint256 private constant FINALITY_THRESHOLD_EXECUTED_INDEX = 144;
    uint256 private constant MESSAGE_BODY_INDEX = 148;

    /// @dev Indices of each field in message body that is extracted from message
    /// Source: https://github.com/circlefin/evm-cctp-contracts/blob/4061786a5726bc05f99fcdb53b0985599f0dbaf7/src/messages/v2/BurnMessageV2.sol#L48-L52
    uint256 private constant BURN_TOKEN_INDEX = 4;
    uint256 private constant MINT_RECIPIENT_INDEX = 36;
    uint256 private constant AMOUNT_INDEX = 68;
    uint256 private constant MAX_FEE_INDEX = 132;
    uint256 private constant FEE_EXECUTED_INDEX = 164;
    uint256 private constant HOOK_DATA_INDEX = 228;

    // Minimum length of the message body (can be longer due to variable actionData)
    uint256 private constant MIN_MSG_BYTES_LENGTH = 728;

    /**
     * @notice Gets the data for the deposit for burn.
     * @param quote The quote that contains the data for the deposit.
     * @return amount The amount of tokens to deposit for burn.
     * @return destinationDomain The destination domain ID for the chain that the tokens are being deposited to.
     * @return mintRecipient The recipent of the minted tokens. This would be the destination periphery contract.
     * @return burnToken The address of the token to burn.
     * @return destinationCaller The address that will call the CCTP receiveMessage function. This would be the destination periphery contract.
     * @return maxFee The maximum fee that can be paid for the deposit.
     * @return minFinalityThreshold The minimum finality threshold for the deposit.
     * @return hookData The hook data for the deposit. Contrains additional data to be used by the destination periphery contract.
     */
    function getDepositForBurnData(
        SponsoredCCTPInterface.SponsoredCCTPQuote memory quote
    )
        internal
        pure
        returns (
            uint256 amount,
            uint32 destinationDomain,
            bytes32 mintRecipient,
            address burnToken,
            bytes32 destinationCaller,
            uint256 maxFee,
            uint32 minFinalityThreshold,
            bytes memory hookData
        )
    {
        amount = quote.amount;
        destinationDomain = quote.destinationDomain;
        mintRecipient = quote.mintRecipient;
        burnToken = quote.burnToken.toAddress();
        destinationCaller = quote.destinationCaller;
        maxFee = quote.maxFee;
        minFinalityThreshold = quote.minFinalityThreshold;
        hookData = abi.encode(
            quote.nonce,
            quote.deadline,
            quote.maxBpsToSponsor,
            quote.maxUserSlippageBps,
            quote.finalRecipient,
            quote.finalToken,
            quote.destinationDex,
            quote.accountCreationMode,
            quote.executionMode,
            quote.actionData
        );
    }

    /**
     * @notice Reads the top-level CCTP message `recipient` (i.e. the contract MessageTransmitter delivers to,
     * typically the TokenMessenger that performs the mint). Assumes `message` has passed length validation.
     */
    function extractRecipient(bytes memory message) internal pure returns (bytes32) {
        return message.toBytes32(RECIPIENT_INDEX);
    }

    /**
     * @notice Reads the CCTP source domain from the message header. Assumes `message` has passed length validation.
     */
    function extractSourceDomain(bytes memory message) internal pure returns (uint32) {
        return message.toUint32(SOURCE_DOMAIN_INDEX);
    }

    /**
     * @notice Reads the burnToken (as bytes32) from the burn message body. Assumes `message` has passed length validation.
     */
    function extractBurnToken(bytes memory message) internal pure returns (bytes32) {
        return message.toBytes32(MESSAGE_BODY_INDEX + BURN_TOKEN_INDEX);
    }

    /**
     * @notice Validates the message that is received from CCTP. If this checks fails, then the quote on source chain was invalid
     * and we are unable to retrieve user's address to send the funds to. In that case the funds will stay in this contract.
     * @param message The message that is received from CCTP.
     * @return isValid True if the message is valid, false otherwise.
     */
    function validateMessage(bytes memory message) internal view returns (bool) {
        // Message must be at least the minimum length (can be longer due to variable actionData)
        if (message.length < MIN_MSG_BYTES_LENGTH) {
            return false;
        }

        // Mint recipient should be this contract
        if (message.toBytes32(MESSAGE_BODY_INDEX + MINT_RECIPIENT_INDEX).toAddress() != address(this)) {
            return false;
        }

        // Validate that finalRecipient and finalToken addresses are valid
        bytes memory messageBody = message.slice(MESSAGE_BODY_INDEX, message.length);
        bytes memory hookData = messageBody.slice(HOOK_DATA_INDEX, messageBody.length);

        // Decode to check address validity
        (, , , , bytes32 finalRecipient, bytes32 finalToken, , , , ) = abi.decode(
            hookData,
            (bytes32, uint256, uint256, uint256, bytes32, bytes32, uint32, uint8, uint8, bytes)
        );

        return finalRecipient.isValidAddress() && finalToken.isValidAddress();
    }

    /**
     * @notice Returns the quote and the fee that was executed from the CCTP message.
     * @param message The message that is received from CCTP.
     * @return quote The quote that contains the data of the deposit.
     * @return feeExecuted The fee that was executed for the deposit. This is the fee that was paid to the CCTP message transmitter.
     */
    function getSponsoredCCTPQuoteData(
        bytes memory message
    ) internal pure returns (SponsoredCCTPInterface.SponsoredCCTPQuote memory quote, uint256 feeExecuted) {
        quote.sourceDomain = message.toUint32(SOURCE_DOMAIN_INDEX);
        quote.destinationDomain = message.toUint32(DESTINATION_DOMAIN_INDEX);
        quote.destinationCaller = message.toBytes32(DESTINATION_CALLER_INDEX);
        quote.minFinalityThreshold = message.toUint32(MIN_FINALITY_THRESHOLD_INDEX);

        // first need to extract the message body from the message
        bytes memory messageBody = message.slice(MESSAGE_BODY_INDEX, message.length);
        quote.mintRecipient = messageBody.toBytes32(MINT_RECIPIENT_INDEX);
        quote.amount = messageBody.toUint256(AMOUNT_INDEX);
        quote.burnToken = messageBody.toBytes32(BURN_TOKEN_INDEX);
        quote.maxFee = messageBody.toUint256(MAX_FEE_INDEX);
        feeExecuted = messageBody.toUint256(FEE_EXECUTED_INDEX);

        bytes memory hookData = messageBody.slice(HOOK_DATA_INDEX, messageBody.length);
        (
            quote.nonce,
            quote.deadline,
            quote.maxBpsToSponsor,
            quote.maxUserSlippageBps,
            quote.finalRecipient,
            quote.finalToken,
            quote.destinationDex,
            quote.accountCreationMode,
            quote.executionMode,
            quote.actionData
        ) = abi.decode(hookData, (bytes32, uint256, uint256, uint256, bytes32, bytes32, uint32, uint8, uint8, bytes));
    }

    /**
     * @notice Validates the signature against the quote.
     * @param signer The signer address that was used to sign the quote.
     * @param quote The quote that contains the data of the deposit.
     * @param signature The signature of the quote.
     * @return isValid True if the signature is valid, false otherwise.
     */
    function validateSignature(
        address signer,
        SponsoredCCTPInterface.SponsoredCCTPQuote memory quote,
        bytes memory signature
    ) internal view returns (bool) {
        // Need to split the hash into two parts to avoid stack too deep error
        bytes32 hash1 = keccak256(
            abi.encode(
                quote.sourceDomain,
                quote.destinationDomain,
                quote.mintRecipient,
                quote.amount,
                quote.burnToken,
                quote.destinationCaller,
                quote.maxFee,
                quote.minFinalityThreshold
            )
        );

        bytes32 hash2 = keccak256(
            abi.encode(
                quote.nonce,
                quote.deadline,
                quote.maxBpsToSponsor,
                quote.maxUserSlippageBps,
                quote.finalRecipient,
                quote.finalToken,
                quote.destinationDex,
                quote.accountCreationMode,
                quote.executionMode,
                keccak256(quote.actionData) // Hash the actionData to keep signature size reasonable
            )
        );

        bytes32 typedDataHash = keccak256(abi.encode(hash1, hash2));
        return SignatureChecker.isValidSignatureNow(signer, typedDataHash, signature);
    }
}


// ===== FILE: contracts/interfaces/SponsoredCCTPInterface.sol =====
//SPDX-License-Identifier: Unlicense
pragma solidity ^0.8.0;

import { SponsoredExecutionModeInterface } from "./SponsoredExecutionModeInterface.sol";

/**
 * @title SponsoredCCTPInterface
 * @notice Interface for the SponsoredCCTP contract
 * @custom:security-contact bugs@across.to
 */
interface SponsoredCCTPInterface is SponsoredExecutionModeInterface {
    // Error thrown when the signature is invalid.
    error InvalidSignature();

    // Error thrown when the nonce is invalid.
    error InvalidNonce();

    // Error thrown when the deadline is invalid.
    error InvalidDeadline();

    // Error thrown when the source domain is invalid.
    error InvalidSourceDomain();

    // Error thrown when the CCTP message transmitter receive message fails.
    error CCTPMessageTransmitterFailed();

    // Error thrown when the amount actually minted into this contract by the CCTP call does not match
    // the `amount - feeExecuted` encoded in the message.
    error InvalidMintedAmount();

    // Error thrown when the CCTP message's top-level `recipient` is not the configured TokenMessenger.
    error InvalidRecipient();

    // Error thrown when the TokenMinter does not link the message's remote burnToken to this periphery's
    // `baseToken` on the local domain.
    error InvalidMintedToken();

    // Error thrown when direct flow destination handler is not a contract.
    error InvalidDirectHandler();

    // Error thrown when the burn token is invalid.
    error InvalidBurnToken();

    event SponsoredDepositForBurn(
        bytes32 indexed quoteNonce,
        address indexed originSender,
        bytes32 indexed finalRecipient,
        uint256 quoteDeadline,
        uint256 maxBpsToSponsor,
        uint256 maxUserSlippageBps,
        bytes32 finalToken,
        uint32 destinationDex,
        uint8 accountCreationMode,
        bytes signature
    );

    event SponsoredCCTPDirectExecution(
        bytes32 indexed quoteNonce,
        address indexed originSender,
        bytes32 indexed finalRecipient,
        uint256 quoteDeadline,
        uint256 maxBpsToSponsor,
        uint256 maxUserSlippageBps,
        bytes32 finalToken,
        uint32 destinationDex,
        uint8 accountCreationMode,
        bytes signature
    );

    // Event when emergency receive is called
    event EmergencyReceiveMessage(bytes32 nonce, address finalRecipent, address finalToken, uint256 amount);

    // Params that will be used to create a sponsored CCTP quote and deposit for burn.
    struct SponsoredCCTPQuote {
        // The domain ID of the source chain.
        uint32 sourceDomain;
        // The domain ID of the destination chain.
        uint32 destinationDomain;
        // The recipient of the minted USDC on the destination chain.
        bytes32 mintRecipient;
        // The amount that the user pays on the source chain.
        uint256 amount;
        // The token that will be burned on the source chain.
        bytes32 burnToken;
        // The caller of the destination chain.
        bytes32 destinationCaller;
        // Maximum fee to pay on the destination domain, specified in units of burnToken
        uint256 maxFee;
        // Minimum finality threshold before allowed to attest
        uint32 minFinalityThreshold;
        // Nonce is used to prevent replay attacks.
        bytes32 nonce;
        // Timestamp of the quote after which it can no longer be used.
        uint256 deadline;
        // The maximum basis points of the amount that can be sponsored.
        uint256 maxBpsToSponsor;
        // Slippage tolerance for the fees on the destination. Used in swap flow, enforced on destination
        uint256 maxUserSlippageBps;
        // The final recipient of the sponsored deposit. This is needed as the mintRecipient will be the
        // handler contract address instead of the final recipient.
        bytes32 finalRecipient;
        // The final token that final recipient will receive. This is needed as it can be different from the burnToken
        // in which case we perform a swap on the destination chain.
        bytes32 finalToken;
        // The destination DEX on HyperCore.
        uint32 destinationDex;
        // AccountCreationMode: Standard or FromUserFunds
        uint8 accountCreationMode;
        // Execution mode: DirectToCore, ArbitraryActionsToCore, or ArbitraryActionsToEVM
        uint8 executionMode;
        // Encoded action data for arbitrary execution. Empty for DirectToCore mode.
        bytes actionData;
    }
}


// ===== FILE: contracts/periphery/mintburn/sponsored-cctp/SponsoredCCTPDstPeriphery.sol =====
//SPDX-License-Identifier: Unlicense
pragma solidity ^0.8.0;

import { BaseModuleHandler } from "../BaseModuleHandler.sol";
import { IMessageTransmitterV2, ITokenMessengerV2 } from "../../../external/interfaces/CCTPInterfaces.sol";
import { SponsoredCCTPQuoteLib } from "../../../libraries/SponsoredCCTPQuoteLib.sol";
import { SponsoredCCTPInterface } from "../../../interfaces/SponsoredCCTPInterface.sol";
import { Bytes32ToAddress } from "../../../libraries/AddressConverters.sol";
import { HyperCoreFlowExecutor } from "../HyperCoreFlowExecutor.sol";
import { ArbitraryEVMFlowExecutor } from "../ArbitraryEVMFlowExecutor.sol";
import { CommonFlowParams, EVMFlowParams, AccountCreationMode as StructsAccountCreationMode } from "../Structs.sol";

import { IERC20Metadata } from "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import { SafeERC20 } from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title SponsoredCCTPDstPeriphery
 * @notice Destination chain periphery contract that supports sponsored/non-sponsored CCTP deposits.
 * @dev This contract is used to receive tokens via CCTP and execute the flow accordingly.
 * @dev IMPORTANT. `BaseModuleHandler` should always be the first contract in inheritance chain. Read 
    `BaseModuleHandler` contract code to learn more.
 */
contract SponsoredCCTPDstPeriphery is BaseModuleHandler, SponsoredCCTPInterface, ArbitraryEVMFlowExecutor {
    using SafeERC20 for IERC20Metadata;
    using Bytes32ToAddress for bytes32;

    /// @notice The CCTP message transmitter contract.
    IMessageTransmitterV2 public immutable cctpMessageTransmitter;

    /// @notice The CCTP token messenger contract that is the expected top-level recipient of MessageTransmitter
    /// deliveries and owns the TokenMinter used to resolve burnToken -> local token for non-direct-deposit flows.
    ITokenMessengerV2 public immutable cctpTokenMessenger;

    /// @notice Base token associated with this handler. The one we receive from the CCTP bridge
    address public immutable baseToken;

    /// @custom:storage-location erc7201:SponsoredCCTPDstPeriphery.main
    struct MainStorage {
        /// @notice The public key of the signer that was used to sign the quotes.
        address signer;
        /// @notice Allow a buffer for quote deadline validation. CCTP transfer might have taken a while to finalize
        uint256 quoteDeadlineBuffer;
        /// @notice A mapping of used nonces to prevent replay attacks.
        mapping(bytes32 => bool) usedNonces;
    }

    // keccak256(abi.encode(uint256(keccak256("erc7201:SponsoredCCTPDstPeriphery.main")) - 1)) & ~bytes32(uint256(0xff));
    bytes32 private constant MAIN_STORAGE_LOCATION = 0xb788edf5b6d001c4df53cb371352fd225afa05a1712075d5f89a08d6b6f79f00;

    function _getMainStorage() private pure returns (MainStorage storage $) {
        assembly {
            $.slot := MAIN_STORAGE_LOCATION
        }
    }

    /**
     * @notice Constructor for the SponsoredCCTPDstPeriphery contract.
     * @param _cctpMessageTransmitter The address of the CCTP message transmitter contract.
     * @param _cctpTokenMessenger The address of the CCTP token messenger contract. Used to pin that a received
     * message actually routed through the real mint flow and that the remote burnToken links to `baseToken`.
     * @param _signer The address of the signer that was used to sign the quotes.
     * @param _donationBox The address of the donation box contract. This is used to store funds that are used for sponsored flows.
     * @param _baseToken The address of the base token which would be the USDC on HyperEVM.
     * @param _multicallHandler The address of the multicall handler contract.
     */
    constructor(
        address _cctpMessageTransmitter,
        address _cctpTokenMessenger,
        address _signer,
        address _donationBox,
        address _baseToken,
        address _multicallHandler
    ) BaseModuleHandler(_donationBox, _baseToken, DEFAULT_ADMIN_ROLE) ArbitraryEVMFlowExecutor(_multicallHandler) {
        baseToken = _baseToken;

        cctpMessageTransmitter = IMessageTransmitterV2(_cctpMessageTransmitter);
        cctpTokenMessenger = ITokenMessengerV2(_cctpTokenMessenger);

        MainStorage storage $ = _getMainStorage();
        $.signer = _signer;
        $.quoteDeadlineBuffer = 30 minutes;

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
    }

    /**
     * @notice Returns the signer address that is used to validate the signatures of the quotes.
     * @return The signer address.
     */
    function signer() external view returns (address) {
        return _getMainStorage().signer;
    }

    /**
     * @notice Returns the quote deadline buffer.
     * @return The quote deadline buffer.
     */
    function quoteDeadlineBuffer() external view returns (uint256) {
        return _getMainStorage().quoteDeadlineBuffer;
    }

    /**
     * @notice Returns true if the nonce has been used, false otherwise.
     * @param nonce The nonce to check.
     * @return True if the nonce has been used, false otherwise.
     */
    function usedNonces(bytes32 nonce) external view returns (bool) {
        return _getMainStorage().usedNonces[nonce];
    }

    /**
     * @notice Sets the signer address that is used to validate the signatures of the quotes.
     * @param _signer The new signer address.
     */
    function setSigner(address _signer) external nonReentrant onlyRole(DEFAULT_ADMIN_ROLE) {
        _getMainStorage().signer = _signer;
    }

    /**
     * @notice Sets the quote deadline buffer. This is used to prevent the quote from being used after it has expired.
     * @param _quoteDeadlineBuffer The new quote deadline buffer.
     */
    function setQuoteDeadlineBuffer(uint256 _quoteDeadlineBuffer) external nonReentrant onlyRole(DEFAULT_ADMIN_ROLE) {
        _getMainStorage().quoteDeadlineBuffer = _quoteDeadlineBuffer;
    }

    /**
     * @notice Emmergency function that can be used to recover funds in cases where it is not possible to go
     * through the normal flow (e.g. HyperEVM <> HyperCore USDC bridge is blacklisted). Receives the message from
     * CCTP and then sends it to final recipient
     * @param message The message that is received from CCTP.
     * @param attestation The attestation that is received from CCTP.
     */
    function emergencyReceiveMessage(
        bytes memory message,
        bytes memory attestation
    ) external nonReentrant onlyRole(PERMISSIONED_BOT_ROLE) {
        uint256 baseBalBefore = IERC20Metadata(baseToken).balanceOf(address(this));
        bool success = cctpMessageTransmitter.receiveMessage(message, attestation);
        if (!success) {
            return;
        }
        // Forward only what the CCTP call actually credited to this contract, so a message that mints nothing
        // (wrong recipient, unlinked burnToken, etc.) cannot drain this contract's prior `baseToken` balance.
        uint256 baseBalAfter = IERC20Metadata(baseToken).balanceOf(address(this));
        uint256 mintedAmount = baseBalAfter - baseBalBefore;

        // Use try-catch to handle potential abi.decode reverts gracefully
        try this.validateMessage(message) returns (bool isValid) {
            if (!isValid) {
                return;
            }
        } catch {
            // Malformed message that causes abi.decode to revert then early return
            return;
        }
        (SponsoredCCTPInterface.SponsoredCCTPQuote memory quote, ) = SponsoredCCTPQuoteLib.getSponsoredCCTPQuoteData(
            message
        );

        address finalRecipient = quote.finalRecipient.toAddress();
        IERC20Metadata(baseToken).safeTransfer(finalRecipient, mintedAmount);

        emit EmergencyReceiveMessage(quote.nonce, finalRecipient, baseToken, mintedAmount);
    }

    /**
     * @notice Receives a message from CCTP and executes the flow accordingly. This function first calls the
     * CCTP message transmitter to receive the funds before validating the quote and executing the flow.
     * @param message The message that is received from CCTP.
     * @param attestation The attestation that is received from CCTP.
     * @param signature The signature of the quote.
     */
    function receiveMessage(
        bytes memory message,
        bytes memory attestation,
        bytes memory signature
    ) external nonReentrant authorizeFundedFlow {
        // Check that the CCTP message was actually routed through our TokenMessenger (which owns the mint flow),
        // and that its TokenMinter links the remote burnToken to our `baseToken` on this domain.
        if (SponsoredCCTPQuoteLib.extractRecipient(message).toAddressUnchecked() != address(cctpTokenMessenger)) {
            revert InvalidRecipient();
        }
        address localToken = cctpTokenMessenger.localMinter().getLocalToken(
            SponsoredCCTPQuoteLib.extractSourceDomain(message),
            SponsoredCCTPQuoteLib.extractBurnToken(message)
        );
        if (localToken != baseToken) {
            revert InvalidMintedToken();
        }

        uint256 baseBalBefore = IERC20Metadata(baseToken).balanceOf(address(this));
        bool success = cctpMessageTransmitter.receiveMessage(message, attestation);
        if (!success) {
            revert CCTPMessageTransmitterFailed();
        }

        // If the hook data is invalid or the mint recipient is not this contract we cannot process the message
        // and therefore we exit. In this case the funds will be kept in this contract.
        // Use try-catch to handle potential abi.decode reverts gracefully
        try this.validateMessage(message) returns (bool isValid) {
            if (!isValid) {
                return;
            }
        } catch {
            // Malformed message that causes abi.decode to revert then early return
            return;
        }
        // Extract the quote and the fee that was executed from the message.
        (SponsoredCCTPInterface.SponsoredCCTPQuote memory quote, uint256 feeExecuted) = SponsoredCCTPQuoteLib
            .getSponsoredCCTPQuoteData(message);

        // Confirm the CCTP call actually minted `baseToken` to this contract.
        uint256 baseBalAfter = IERC20Metadata(baseToken).balanceOf(address(this));
        uint256 amountAfterFees = quote.amount - feeExecuted;
        if (baseBalAfter - baseBalBefore != amountAfterFees) {
            revert InvalidMintedAmount();
        }

        // Validate the quote and the signature. Revert on invalid to prevent griefing attacks
        // where an attacker provides correct message/attestation but invalid signature.
        _validateQuoteOrRevert(quote, signature);

        _executeQuote(quote, feeExecuted);
    }

    /**
     * @notice Direct execution entrypoint for same-chain flows that bypass CCTP transport.
     * @dev Caller must hold DIRECT_CALLER_ROLE and transfer baseToken funds to this contract before invoking.
     * @param quote The quote that contains the data for the deposit.
     */
    function directReceiveMessage(
        SponsoredCCTPInterface.SponsoredCCTPQuote memory quote
    ) external nonReentrant authorizeFundedFlow onlyRole(DIRECT_CALLER_ROLE) {
        if (quote.burnToken.toAddress() != baseToken) revert InvalidBurnToken();
        MainStorage storage $ = _getMainStorage();
        if ($.usedNonces[quote.nonce]) revert InvalidNonce();
        $.usedNonces[quote.nonce] = true;
        _executeQuote(quote, 0);
    }

    /**
     * @notice Shared execution logic for both CCTP-bridged and direct flows.
     * @param quote The validated quote.
     * @param feeExecuted The CCTP fee deducted (0 for direct flows).
     */
    function _executeQuote(SponsoredCCTPInterface.SponsoredCCTPQuote memory quote, uint256 feeExecuted) internal {
        uint256 amountAfterFees = quote.amount - feeExecuted;

        CommonFlowParams memory commonParams = CommonFlowParams({
            amountInEVM: amountAfterFees,
            quoteNonce: quote.nonce,
            finalRecipient: quote.finalRecipient.toAddress(),
            finalToken: quote.finalToken.toAddress(),
            destinationDex: quote.destinationDex,
            maxBpsToSponsor: quote.maxBpsToSponsor,
            extraFeesIncurred: feeExecuted,
            accountCreationMode: StructsAccountCreationMode(quote.accountCreationMode)
        });

        // Route to appropriate execution based on executionMode
        if (
            quote.executionMode == uint8(ExecutionMode.ArbitraryActionsToCore) ||
            quote.executionMode == uint8(ExecutionMode.ArbitraryActionsToEVM)
        ) {
            // Execute flow with arbitrary evm actions
            _executeWithEVMFlow(
                EVMFlowParams({
                    commonParams: commonParams,
                    initialToken: baseToken,
                    actionData: quote.actionData,
                    transferToCore: quote.executionMode == uint8(ExecutionMode.ArbitraryActionsToCore)
                })
            );
        } else {
            // Execute standard HyperCore flow (default) via delegatecall
            _delegateToHyperCore(
                abi.encodeCall(HyperCoreFlowExecutor.executeFlow, (commonParams, quote.maxUserSlippageBps))
            );
        }
    }

    /**
     * @notice External wrapper for validateMessage to enable try-catch for safe abi.decode handling
     * @param message The CCTP message to validate
     * @return True if the message is valid, false otherwise
     */
    function validateMessage(bytes memory message) external view returns (bool) {
        return SponsoredCCTPQuoteLib.validateMessage(message);
    }

    /**
     * @notice Validates the quote and signature, reverting with specific errors if invalid.
     * @dev This prevents griefing attacks where an attacker provides correct message/attestation
     * but an invalid signature, which would otherwise cause loss of sponsorship and custom actions.
     * @param quote The quote to validate.
     * @param signature The signature to validate against the quote.
     */
    function _validateQuoteOrRevert(
        SponsoredCCTPInterface.SponsoredCCTPQuote memory quote,
        bytes memory signature
    ) internal {
        MainStorage storage $ = _getMainStorage();

        if (!SponsoredCCTPQuoteLib.validateSignature($.signer, quote, signature)) {
            revert InvalidSignature();
        }
        if ($.usedNonces[quote.nonce]) {
            revert InvalidNonce();
        }
        if (quote.deadline + $.quoteDeadlineBuffer < block.timestamp) {
            revert InvalidDeadline();
        }

        $.usedNonces[quote.nonce] = true;
    }

    function _executeWithEVMFlow(EVMFlowParams memory params) internal {
        params.commonParams = ArbitraryEVMFlowExecutor._executeFlow(params);

        // If the full balance was deposited as a part of EVM action, return early
        if (params.commonParams.amountInEVM == 0) {
            return;
        }

        // Route to appropriate destination based on transferToCore flag
        _delegateToHyperCore(
            params.transferToCore
                ? abi.encodeCall(HyperCoreFlowExecutor.executeSimpleTransferFlow, (params.commonParams))
                : abi.encodeCall(HyperCoreFlowExecutor.fallbackHyperEVMFlow, (params.commonParams))
        );
    }
}


// ===== FILE: contracts/libraries/AddressConverters.sol =====
// SPDX-License-Identifier: BUSL-1.1
pragma solidity ^0.8.0;

library Bytes32ToAddress {
    /**************************************
     *              ERRORS                *
     **************************************/
    error InvalidBytes32();

    function toAddress(bytes32 _bytes32) internal pure returns (address) {
        checkAddress(_bytes32);
        return address(uint160(uint256(_bytes32)));
    }

    function toAddressUnchecked(bytes32 _bytes32) internal pure returns (address) {
        return address(uint160(uint256(_bytes32)));
    }

    function checkAddress(bytes32 _bytes32) internal pure {
        if (!isValidAddress(_bytes32)) {
            revert InvalidBytes32();
        }
    }

    function isValidAddress(bytes32 _bytes32) internal pure returns (bool) {
        return uint256(_bytes32) >> 160 == 0;
    }
}

library AddressToBytes32 {
    function toBytes32(address _address) internal pure returns (bytes32) {
        return bytes32(uint256(uint160(_address)));
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts-v4/utils/Context.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.4) (utils/Context.sol)

pragma solidity ^0.8.0;

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


// ===== FILE: node_modules/_openzeppelin/contracts-v4/token/ERC20/extensions/IERC20Permit.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.4) (token/ERC20/extensions/IERC20Permit.sol)

pragma solidity ^0.8.0;

/**
 * @dev Interface of the ERC20 Permit extension allowing approvals to be made via signatures, as defined in
 * https://eips.ethereum.org/EIPS/eip-2612[EIP-2612].
 *
 * Adds the {permit} method, which can be used to change an account's ERC20 allowance (see {IERC20-allowance}) by
 * presenting a message signed by the account. By not relying on {IERC20-approve}, the token holder account doesn't
 * need to send a transaction, and thus is not required to hold Ether at all.
 *
 * ==== Security Considerations
 *
 * There are two important considerations concerning the use of `permit`. The first is that a valid permit signature
 * expresses an allowance, and it should not be assumed to convey additional meaning. In particular, it should not be
 * considered as an intention to spend the allowance in any specific way. The second is that because permits have
 * built-in replay protection and can be submitted by anyone, they can be frontrun. A protocol that uses permits should
 * take this into consideration and allow a `permit` call to fail. Combining these two aspects, a pattern that may be
 * generally recommended is:
 *
 * ```solidity
 * function doThingWithPermit(..., uint256 value, uint256 deadline, uint8 v, bytes32 r, bytes32 s) public {
 *     try token.permit(msg.sender, address(this), value, deadline, v, r, s) {} catch {}
 *     doThing(..., value);
 * }
 *
 * function doThing(..., uint256 value) public {
 *     token.safeTransferFrom(msg.sender, address(this), value);
 *     ...
 * }
 * ```
 *
 * Observe that: 1) `msg.sender` is used as the owner, leaving no ambiguity as to the signer intent, and 2) the use of
 * `try/catch` allows the permit to fail and makes the code tolerant to frontrunning. (See also
 * {SafeERC20-safeTransferFrom}).
 *
 * Additionally, note that smart contract wallets (such as Argent or Safe) are not able to produce permit signatures, so
 * contracts should have entry points that don't rely on permit.
 */
interface IERC20Permit {
    /**
     * @dev Sets `value` as the allowance of `spender` over ``owner``'s tokens,
     * given ``owner``'s signed approval.
     *
     * IMPORTANT: The same issues {IERC20-approve} has related to transaction
     * ordering also apply here.
     *
     * Emits an {Approval} event.
     *
     * Requirements:
     *
     * - `spender` cannot be the zero address.
     * - `deadline` must be a timestamp in the future.
     * - `v`, `r` and `s` must be a valid `secp256k1` signature from `owner`
     * over the EIP712-formatted function arguments.
     * - the signature must use ``owner``'s current nonce (see {nonces}).
     *
     * For more information on the signature format, see the
     * https://eips.ethereum.org/EIPS/eip-2612#specification[relevant EIP
     * section].
     *
     * CAUTION: See Security Considerations above.
     */
    function permit(
        address owner,
        address spender,
        uint256 value,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external;

    /**
     * @dev Returns the current nonce for `owner`. This value must be
     * included whenever a signature is generated for {permit}.
     *
     * Every successful call to {permit} increases ``owner``'s nonce by one. This
     * prevents a signature from being used multiple times.
     */
    function nonces(address owner) external view returns (uint256);

    /**
     * @dev Returns the domain separator used in the encoding of the signature for {permit}, as defined by {EIP712}.
     */
    // solhint-disable-next-line func-name-mixedcase
    function DOMAIN_SEPARATOR() external view returns (bytes32);
}


// ===== FILE: node_modules/_openzeppelin/contracts-v4/utils/cryptography/SignatureChecker.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.0) (utils/cryptography/SignatureChecker.sol)

pragma solidity ^0.8.0;

import "./ECDSA.sol";
import "../../interfaces/IERC1271.sol";

/**
 * @dev Signature verification helper that can be used instead of `ECDSA.recover` to seamlessly support both ECDSA
 * signatures from externally owned accounts (EOAs) as well as ERC1271 signatures from smart contract wallets like
 * Argent and Gnosis Safe.
 *
 * _Available since v4.1._
 */
library SignatureChecker {
    /**
     * @dev Checks if a signature is valid for a given signer and data hash. If the signer is a smart contract, the
     * signature is validated against that smart contract using ERC1271, otherwise it's validated using `ECDSA.recover`.
     *
     * NOTE: Unlike ECDSA signatures, contract signatures are revocable, and the outcome of this function can thus
     * change through time. It could return true at block N and false at block N+1 (or the opposite).
     */
    function isValidSignatureNow(address signer, bytes32 hash, bytes memory signature) internal view returns (bool) {
        (address recovered, ECDSA.RecoverError error) = ECDSA.tryRecover(hash, signature);
        return
            (error == ECDSA.RecoverError.NoError && recovered == signer) ||
            isValidERC1271SignatureNow(signer, hash, signature);
    }

    /**
     * @dev Checks if a signature is valid for a given signer and data hash. The signature is validated
     * against the signer smart contract using ERC1271.
     *
     * NOTE: Unlike ECDSA signatures, contract signatures are revocable, and the outcome of this function can thus
     * change through time. It could return true at block N and false at block N+1 (or the opposite).
     */
    function isValidERC1271SignatureNow(
        address signer,
        bytes32 hash,
        bytes memory signature
    ) internal view returns (bool) {
        (bool success, bytes memory result) = signer.staticcall(
            abi.encodeWithSelector(IERC1271.isValidSignature.selector, hash, signature)
        );
        return (success &&
            result.length >= 32 &&
            abi.decode(result, (bytes32)) == bytes32(IERC1271.isValidSignature.selector));
    }
}


// ===== FILE: contracts/external/libraries/BytesLib.sol =====
// SPDX-License-Identifier: BUSL-1.1
pragma solidity ^0.8.0;

import { Bytes } from "@openzeppelin/contracts/utils/Bytes.sol";

library BytesLib {
    /**************************************
     *              ERRORS                *
     **************************************/
    error OutOfBounds();

    /**************************************
     *              FUNCTIONS              *
     **************************************/

    // The following 4 functions are copied from solidity-bytes-utils library
    // https://github.com/GNSPS/solidity-bytes-utils/blob/fc502455bb2a7e26a743378df042612dd50d1eb9/contracts/BytesLib.sol#L323C5-L398C6
    // Code was copied, and slightly modified to use revert instead of require

    /**
     * @notice Reads a uint8 from a bytes array at a given start index (for tightly packed data)
     * @param _bytes The bytes array to convert
     * @param _start The start index of the uint8
     * @return result The uint8 result
     */
    function toUint8(bytes memory _bytes, uint256 _start) internal pure returns (uint8) {
        if (_bytes.length < _start + 1) {
            revert OutOfBounds();
        }
        uint8 tempUint;

        assembly {
            tempUint := mload(add(add(_bytes, 0x1), _start))
        }

        return tempUint;
    }

    /**
     * @notice Reads a uint16 from a bytes array at a given start index (for tightly packed data)
     * @notice Reads a uint16 from a bytes array at a given start index
     * @param _bytes The bytes array to convert
     * @param _start The start index of the uint16
     * @return result The uint16 result
     */
    function toUint16(bytes memory _bytes, uint256 _start) internal pure returns (uint16 result) {
        if (_bytes.length < _start + 2) {
            revert OutOfBounds();
        }

        // solhint-disable-next-line no-inline-assembly
        assembly {
            result := mload(add(add(_bytes, 0x2), _start))
        }
    }

    /**
     * @notice Reads a uint32 from a bytes array at a given start index (for tightly packed data)
     * @notice Reads a uint32 from a bytes array at a given start index
     * @param _bytes The bytes array to convert
     * @param _start The start index of the uint32
     * @return result The uint32 result
     */
    function toUint32(bytes memory _bytes, uint256 _start) internal pure returns (uint32 result) {
        if (_bytes.length < _start + 4) {
            revert OutOfBounds();
        }

        // solhint-disable-next-line no-inline-assembly
        assembly {
            result := mload(add(add(_bytes, 0x4), _start))
        }
    }

    /**
     * @notice Reads a uint256 from a bytes array at a given start index
     * @param _bytes The bytes array to convert
     * @param _start The start index of the uint256
     * @return result The uint256 result
     */
    function toUint256(bytes memory _bytes, uint256 _start) internal pure returns (uint256 result) {
        if (_bytes.length < _start + 32) {
            revert OutOfBounds();
        }

        // solhint-disable-next-line no-inline-assembly
        assembly {
            result := mload(add(add(_bytes, 0x20), _start))
        }
    }

    /**
     * @notice Reads a bytes32 from a bytes array at a given start index
     * @param _bytes The bytes array to convert
     * @param _start The start index of the bytes32
     * @return result The bytes32 result
     */
    function toBytes32(bytes memory _bytes, uint256 _start) internal pure returns (bytes32 result) {
        if (_bytes.length < _start + 32) {
            revert OutOfBounds();
        }

        // solhint-disable-next-line no-inline-assembly
        assembly {
            result := mload(add(add(_bytes, 0x20), _start))
        }
    }

    /**
     * @notice Reads a bytes array from a bytes array at a given start index and length
     * Source: OpenZeppelin Contracts v5 (utils/Bytes.sol)
     * @param _bytes The bytes array to convert
     * @param _start The start index of the bytes array
     * @param _end The end index of the bytes array
     * @return result The bytes array result
     */
    function slice(bytes memory _bytes, uint256 _start, uint256 _end) internal pure returns (bytes memory result) {
        return Bytes.slice(_bytes, _start, _end);
    }
}


// ===== FILE: contracts/interfaces/SponsoredExecutionModeInterface.sol =====
// SPDX-License-Identifier: BUSL-1.1
pragma solidity ^0.8.0;

/**
 * @notice Shared execution modes for sponsored bridging flows.
 */
interface SponsoredExecutionModeInterface {
    // Send to core and perform swap (if needed) there.
    enum ExecutionMode {
        DirectToCore,
        // Execute arbitrary actions (like a swap) on HyperEVM, then transfer to HyperCore.
        ArbitraryActionsToCore,
        // Execute arbitrary actions on HyperEVM only (no HyperCore transfer).
        ArbitraryActionsToEVM
    }
}


// ===== FILE: contracts/periphery/mintburn/BaseModuleHandler.sol =====
//SPDX-License-Identifier: Unlicense
pragma solidity ^0.8.0;

import { AuthorizedFundedFlow } from "./AuthorizedFundedFlow.sol";
import { HyperCoreFlowExecutor } from "./HyperCoreFlowExecutor.sol";
import { HyperCoreFlowRoles } from "./HyperCoreFlowRoles.sol";

// Note: v5 is necessary since v4 does not use ERC-7201.
import { AccessControlUpgradeable } from "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import { ReentrancyGuard } from "@openzeppelin/contracts-v4/security/ReentrancyGuard.sol";

/**
 * @notice Base contract for module handlers that use delegatecall to interact with HyperCoreFlowExecutor
 * @dev Uses AccessControlUpgradeable to ensure storage compatibility with HyperCoreFlowExecutor when using delegatecall
 */
abstract contract BaseModuleHandler is
    AccessControlUpgradeable,
    ReentrancyGuard,
    AuthorizedFundedFlow,
    HyperCoreFlowRoles
{
    /// @notice Address of the underlying hypercore module
    address public immutable hyperCoreModule;

    constructor(address _donationBox, address _baseToken, bytes32 _roleAdmin) {
        hyperCoreModule = address(new HyperCoreFlowExecutor(_donationBox, _baseToken));

        _setRoleAdmin(PERMISSIONED_BOT_ROLE, _roleAdmin);
        _setRoleAdmin(FUNDS_SWEEPER_ROLE, _roleAdmin);
        _setRoleAdmin(DIRECT_CALLER_ROLE, _roleAdmin);
    }

    /// @notice Fallback function to proxy all calls to the HyperCore module via delegatecall
    /// @dev Permissioning is enforced by the delegated function's own modifiers (e.g. onlyPermissionedBot)
    fallback(bytes calldata data) external nonReentrant returns (bytes memory) {
        return _delegateToHyperCore(data);
    }

    /// @notice Internal delegatecall helper
    function _delegateToHyperCore(bytes memory data) internal returns (bytes memory) {
        (bool success, bytes memory ret) = hyperCoreModule.delegatecall(data);
        if (!success) {
            assembly {
                revert(add(ret, 32), mload(ret))
            }
        }
        return ret;
    }
}


// ===== FILE: contracts/periphery/mintburn/HyperCoreFlowExecutor.sol =====
//SPDX-License-Identifier: Unlicense
pragma solidity ^0.8.0;

import { AuthorizedFundedFlow } from "./AuthorizedFundedFlow.sol";
import { HyperCoreFlowRoles } from "./HyperCoreFlowRoles.sol";
import { DonationBox } from "../../chain-adapters/DonationBox.sol";
import { HyperCoreLib, CoreTokenInfo } from "../../libraries/HyperCoreLib.sol";
import { FinalTokenInfo, CommonFlowParams, AccountCreationMode } from "./Structs.sol";
import { SwapHandler } from "./SwapHandler.sol";
import { BPS_SCALAR, BPS_DECIMALS } from "./Constants.sol";

// Note: v5 is necessary since v4 does not use ERC-7201.
import { AccessControlUpgradeable } from "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import { IERC20 } from "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import { SafeERC20 } from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title HyperCoreFlowExecutor
 * @notice Contract handling HyperCore interactions for transfer-to-core or swap-with-core actions after stablecoin bridge transactions
 * @dev This contract is designed to work with stablecoins. baseToken and every finalToken should all be stablecoins.
 *
 * @dev This contract is intended to be used exclusively via delegatecall from handler contracts.
 * Direct calls to this contract will produce incorrect results because functions rely on the
 * caller's context, including address(this) for calculations and storage layout from the
 * delegating contract.
 *
 * @custom:security-contact bugs@across.to
 */
contract HyperCoreFlowExecutor is AccessControlUpgradeable, AuthorizedFundedFlow, HyperCoreFlowRoles {
    using SafeERC20 for IERC20;

    // Common decimals scalars
    uint256 public constant PPM_DECIMALS = 6;
    uint256 public constant PPM_SCALAR = 10 ** PPM_DECIMALS;
    uint64 public constant ONEX1e8 = 10 ** 8;

    /// @notice The donation box contract.
    DonationBox public immutable donationBox;

    /// @notice All operations performed in this contract are relative to this baseToken
    address public immutable baseToken;

    /// @notice A struct used for storing state of a swap flow that has been initialized, but not yet finished
    struct SwapFlowState {
        address finalRecipient;
        address finalToken;
        uint32 destinationDex;
        uint64 minAmountToSend; // for sponsored: one to one, non-sponsored: one to one minus slippage
        uint64 maxAmountToSend; // for sponsored: one to one (from total bridged amt), for non-sponsored: one to one, less bridging fees incurred
        bool isSponsored;
        bool finalized;
    }

    /// @custom:storage-location erc7201:HyperCoreFlowExecutor.main
    struct MainStorage {
        /// @notice A mapping of token addresses to their core token info.
        mapping(address => CoreTokenInfo) coreTokenInfos;
        /// @notice A mapping of token address to additional relevan info for final tokens, like Hyperliquid market params
        mapping(address => FinalTokenInfo) finalTokenInfos;
        /// @notice The block number of the last funds pull action per final token: either as a part of finalizing pending swaps,
        /// or an admin funds pull
        mapping(address finalToken => uint256 lastPullFundsBlock) lastPullFundsBlock;
        /// @notice A mapping containing the pending state between initializing the swap flow and finalizing it
        mapping(bytes32 quoteNonce => SwapFlowState swap) swaps;
        /// @notice The cumulative amount of funds sponsored for each final token.
        mapping(address => uint256) cumulativeSponsoredAmount;
        /// @notice The cumulative amount of activation fees sponsored for each final token.
        mapping(address => uint256) cumulativeSponsoredActivationFee;
    }

    // keccak256(abi.encode(uint256(keccak256("erc7201:HyperCoreFlowExecutor.main")) - 1)) & ~bytes32(uint256(0xff));
    bytes32 private constant MAIN_STORAGE_LOCATION = 0x6c70e510d36398bee89cc6e19ea6807a9915863d7d724712e0b3c15b01368b00;

    function _getMainStorage() private pure returns (MainStorage storage $) {
        assembly {
            $.slot := MAIN_STORAGE_LOCATION
        }
    }

    /**************************************
     *            EVENTS               *
     **************************************/

    /**
     * @notice Emitted when the donation box is insufficient funds.
     * @param quoteNonce Unique identifier for this quote/transaction
     * @param token The token address that was requested
     * @param amount The amount requested from the donation box
     * @param balance The actual balance available in the donation box
     */
    event DonationBoxInsufficientFunds(bytes32 indexed quoteNonce, address token, uint256 amount, uint256 balance);

    /**
     * @notice Emitted whenever the account is not activated in the non-sponsored flow. We fall back to HyperEVM flow in that case
     * @param quoteNonce Unique identifier for this quote/transaction
     * @param user The address of the user whose account is not activated
     */
    event AccountNotActivated(bytes32 indexed quoteNonce, address user);

    /**
     * @notice Emitted when a simple transfer to core is executed.
     * @param quoteNonce Unique identifier for this quote/transaction
     * @param finalRecipient The address receiving the funds on HyperCore
     * @param finalToken The token address being transferred
     * @param evmAmountIn The amount received on HyperEVM (in finalToken)
     * @param bridgingFeesIncurred The bridging fees incurred (in finalToken)
     * @param evmAmountSponsored The amount sponsored from the donation box (in finalToken)
     */
    event SimpleTransferFlowCompleted(
        bytes32 indexed quoteNonce,
        address indexed finalRecipient,
        address indexed finalToken,
        // All amounts are in finalToken
        uint256 evmAmountIn,
        uint256 bridgingFeesIncurred,
        uint256 evmAmountSponsored
    );

    /**
     * @notice Emitted upon successful completion of fallback HyperEVM flow
     * @param quoteNonce Unique identifier for this quote/transaction
     * @param finalRecipient The address receiving the funds on HyperEVM
     * @param finalToken The token address being transferred
     * @param evmAmountIn The amount received on HyperEVM (in finalToken)
     * @param bridgingFeesIncurred The bridging fees incurred (in finalToken)
     * @param evmAmountSponsored The amount sponsored from the donation box (in finalToken)
     */
    event FallbackHyperEVMFlowCompleted(
        bytes32 indexed quoteNonce,
        address indexed finalRecipient,
        address indexed finalToken,
        // All amounts are in finalToken
        uint256 evmAmountIn,
        uint256 bridgingFeesIncurred,
        uint256 evmAmountSponsored
    );

    /**
     * @notice Emitted when a swap flow is initialized
     * @param quoteNonce Unique identifier for this quote/transaction
     * @param finalRecipient The address that will receive the swapped funds on HyperCore
     * @param finalToken The token address to swap to
     * @param evmAmountIn The amount received on HyperEVM (in baseToken)
     * @param bridgingFeesIncurred The bridging fees incurred (in baseToken)
     * @param coreAmountIn The amount sent to HyperCore (in finalToken)
     * @param minAmountToSend Minimum amount to send to user after swap (in finalToken)
     * @param maxAmountToSend Maximum amount to send to user after swap (in finalToken)
     */
    event SwapFlowInitialized(
        bytes32 indexed quoteNonce,
        address indexed finalRecipient,
        address indexed finalToken,
        // In baseToken
        uint256 evmAmountIn,
        uint256 bridgingFeesIncurred,
        // In finalToken
        uint256 coreAmountIn,
        uint64 minAmountToSend,
        uint64 maxAmountToSend
    );

    /**
     * @notice Emitted when a swap flow is finalized
     * @param quoteNonce Unique identifier for this quote/transaction
     * @param finalRecipient The address that received the swapped funds on HyperCore
     * @param finalToken The token address that was swapped to
     * @param totalSent Total amount sent to the final recipient on HyperCore (in finalToken)
     * @param evmAmountSponsored The amount sponsored from the donation box (in EVM finalToken)
     */
    event SwapFlowFinalized(
        bytes32 indexed quoteNonce,
        address indexed finalRecipient,
        address indexed finalToken,
        // In finalToken
        uint64 totalSent,
        // In EVM finalToken
        uint256 evmAmountSponsored
    );

    /**
     * @notice Emitted upon cancelling a Limit order
     * @param token The token address for which the limit order was placed
     * @param cloid Client order ID of the cancelled limit order
     */
    event CancelledLimitOrder(address indexed token, uint128 indexed cloid);

    /**
     * @notice Emitted upon submitting a Limit order
     * @param token The token address for which the limit order is placed
     * @param priceX1e8 The limit order price (scaled by 1e8)
     * @param sizeX1e8 The limit order size (scaled by 1e8)
     * @param cloid Client order ID of the submitted limit order
     */
    event SubmittedLimitOrder(address indexed token, uint64 priceX1e8, uint64 sizeX1e8, uint128 indexed cloid);

    /**
     * @notice Emitted when we have to fall back from the swap flow because it's too expensive (either to sponsor or the slippage is too big)
     * @param quoteNonce Unique identifier for this quote/transaction
     * @param finalToken The token address that was intended to be swapped to
     * @param estBpsSlippage Estimated slippage in basis points
     * @param maxAllowableBpsSlippage Maximum allowable slippage in basis points
     */
    event SwapFlowTooExpensive(
        bytes32 indexed quoteNonce,
        address indexed finalToken,
        uint256 estBpsSlippage,
        uint256 maxAllowableBpsSlippage
    );

    /**
     * @notice Emitted when we can't bridge some token from HyperEVM to HyperCore
     * @param quoteNonce Unique identifier for this quote/transaction
     * @param token The token address that is unsafe to bridge
     * @param amount The amount that was attempted to be bridged
     */
    event UnsafeToBridge(bytes32 indexed quoteNonce, address indexed token, uint64 amount);

    /**
     * @notice Emitted whenever donationBox funds are used for activating a user account
     * @param quoteNonce Unique identifier for this quote/transaction
     * @param finalRecipient The address of the user whose account is being activated
     * @param fundingToken The token used to fund the account activation
     * @param evmAmountSponsored The amount sponsored for activation (in EVM token)
     */
    event SponsoredAccountActivation(
        bytes32 indexed quoteNonce,
        address indexed finalRecipient,
        address indexed fundingToken,
        uint256 evmAmountSponsored
    );

    /**
     * @notice Emitted whenever a user account is activated using funds from the user's transfer/swap
     * @param quoteNonce Unique identifier for this quote/transaction
     * @param user The address of the user whose account is being activated
     * @param token The token used to fund the account activation
     * @param amountCore The amount paid for activation (in Core token units)
     */
    event AccountActivatedFromUserFunds(
        bytes32 indexed quoteNonce,
        address indexed user,
        address indexed token,
        uint64 amountCore
    );

    /**
     * @notice Emitted whenever a new CoreTokenInfo is configured
     * @param token The token address being configured
     * @param coreIndex The index of the token on HyperCore
     * @param canBeUsedForAccountActivation Whether this token can be used to pay for account activation
     * @param accountActivationFeeCore The account activation fee amount (in Core token units)
     * @param bridgeSafetyBufferCore The safety buffer for bridging (in Core token units)
     */
    event SetCoreTokenInfo(
        address indexed token,
        uint32 coreIndex,
        bool canBeUsedForAccountActivation,
        uint64 accountActivationFeeCore,
        uint64 bridgeSafetyBufferCore
    );

    /// @notice Emitted whenever a new FinalTokenInfo is configured
    event SetFinalTokenInfo(
        address indexed token,
        uint32 spotIndex,
        bool isBuy,
        uint32 feePpm,
        address indexed swapHandler,
        uint32 suggestedFeeDiscountBps
    );

    /**
     * @notice Emitted when we do an ad-hoc send of sponsorship funds to one of the Swap Handlers
     * @param token The token address being sent to the swap handler
     * @param evmAmountSponsored The amount sponsored from the donation box (in EVM token)
     */
    event SentSponsorshipFundsToSwapHandler(address indexed token, uint256 evmAmountSponsored);

    /**************************************
     *            ERRORS               *
     **************************************/

    /// @notice Thrown when an attempt to finalize a non-existing swap is made
    error SwapDoesNotExist();

    /// @notice Thrown when an attemp to finalize an already finalized swap is made
    error SwapAlreadyFinalized();

    /// @notice Thrown when trying to finalize a quoteNonce, calling a finalizeSwapFlows with an incorrect token
    error WrongSwapFinalizationToken(bytes32 quoteNonce);

    /// @notice Emitted when we're inside the sponsored flow and a user doesn't have a HyperCore account activated. The
    /// bot should activate user's account first by calling `activateUserAccount`
    error AccountNotActivatedError(address user);

    /// @notice Thrown when we can't bridge some token from HyperEVM to HyperCore
    error UnsafeToBridgeError(address token, uint64 amount);

    /// @notice Thrown when finalizing a swap flow, if a finalToken used to be eligible for account activation, but is no longer
    error TokenNotEligibleForActivation();

    /// @notice Thrown when a user order total is not enough to cover account activation when finalizing a swap flow
    error InsufficientFundsForActivation();

    /**************************************
     *            MODIFIERS               *
     **************************************/

    modifier onlyExistingCoreToken(address evmTokenAddress) {
        _getExistingCoreTokenInfo(evmTokenAddress);
        _;
    }

    /// @notice Reverts if the token is not configured
    function _getExistingCoreTokenInfo(
        address evmTokenAddress
    ) internal view returns (CoreTokenInfo memory coreTokenInfo) {
        coreTokenInfo = _getMainStorage().coreTokenInfos[evmTokenAddress];
        require(
            coreTokenInfo.tokenInfo.evmContract != address(0) && coreTokenInfo.tokenInfo.weiDecimals != 0,
            "CoreTokenInfo not set"
        );
    }

    /// @notice Reverts if the token is not configured
    function _getExistingFinalTokenInfo(
        address evmTokenAddress
    ) internal view returns (FinalTokenInfo memory finalTokenInfo) {
        finalTokenInfo = _getMainStorage().finalTokenInfos[evmTokenAddress];
        require(address(finalTokenInfo.swapHandler) != address(0), "FinalTokenInfo not set");
    }

    /**
     *
     * @param _donationBox Sponsorship funds live here
     * @param _baseToken Main token used with this Forwarder
     */
    constructor(address _donationBox, address _baseToken) {
        // Set immutable variables only
        donationBox = DonationBox(_donationBox);
        baseToken = _baseToken;
    }

    /****************************************
     *            VIEW FUNCTIONS           *
     **************************************/

    /**
     * @notice Returns the core token info for a given token address.
     * @param token The token address.
     * @return The core token info for the given token address.
     */
    function coreTokenInfos(address token) external view returns (CoreTokenInfo memory) {
        return _getMainStorage().coreTokenInfos[token];
    }

    /**
     * @notice Returns the final token info for a given token address.
     * @param token The token address.
     * @return The final token info for the given token address.
     */
    function finalTokenInfos(address token) external view returns (FinalTokenInfo memory) {
        return _getMainStorage().finalTokenInfos[token];
    }

    /**
     * @notice Returns the block number of the last time funds were pulled from the donation box.
     * @param token The token address.
     * @return The block number of the last time funds were pulled from the donation box for the given token address.
     */
    function lastPullFundsBlock(address token) external view returns (uint256) {
        return _getMainStorage().lastPullFundsBlock[token];
    }

    /**
     * @notice Returns the swap info for a given quote nonce.
     * @param quoteNonce The quote nonce.
     * @return The swap info for the given quote nonce.
     */
    function swaps(bytes32 quoteNonce) external view returns (SwapFlowState memory) {
        return _getMainStorage().swaps[quoteNonce];
    }

    /**
     * @notice Returns the cumulative sponsored amount for a given token address.
     * @param token The token address.
     * @return The cumulative sponsored amount for the given token address.
     */
    function cumulativeSponsoredAmount(address token) external view returns (uint256) {
        return _getMainStorage().cumulativeSponsoredAmount[token];
    }

    /**
     * @notice Returns the cumulative sponsored activation fee for a given token address.
     * @param token The token address.
     * @return The cumulative sponsored activation fee for the given token address.
     */
    function cumulativeSponsoredActivationFee(address token) external view returns (uint256) {
        return _getMainStorage().cumulativeSponsoredActivationFee[token];
    }

    /**************************************
     *      CONFIGURATION FUNCTIONS       *
     **************************************/

    /**
     * @notice Set or update information for the token to use it in this contract
     * @dev To be able to use the token in the swap flow, FinalTokenInfo has to be set as well
     * @dev Setting core token info to incorrect values can lead to loss of funds. Should NEVER be unset while the
     * finalTokenParams are not unset
     * @param token The token address being configured
     * @param coreIndex The index of the token on HyperCore
     * @param canBeUsedForAccountActivation Whether this token can be used to pay for account activation
     * @param accountActivationFeeCore The account activation fee amount (in Core token units)
     * @param bridgeSafetyBufferCore The safety buffer for bridging (in Core token units)
     */
    function setCoreTokenInfo(
        address token,
        uint32 coreIndex,
        bool canBeUsedForAccountActivation,
        uint64 accountActivationFeeCore,
        uint64 bridgeSafetyBufferCore
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        _setCoreTokenInfo(
            token,
            coreIndex,
            canBeUsedForAccountActivation,
            accountActivationFeeCore,
            bridgeSafetyBufferCore
        );
    }

    /**
     * @notice Sets the parameters for a final token.
     * @dev This function deploys a new SwapHandler contract if one is not already set. If the final token
     * can't be used for account activation, the handler will be left unactivated and would need to be activated by the caller.
     * @param finalToken The address of the final token.
     * @param spotIndex The index of the asset in the Hyperliquid market.
     * @param isBuy Whether the final token is a buy or a sell.
     * @param feePpm The fee in parts per million.
     * @param suggestedDiscountBps The suggested slippage in basis points.
     */
    function setFinalTokenInfo(
        address finalToken,
        uint32 spotIndex,
        bool isBuy,
        uint32 feePpm,
        uint32 suggestedDiscountBps
    ) external onlyExistingCoreToken(finalToken) onlyRole(DEFAULT_ADMIN_ROLE) {
        MainStorage storage $ = _getMainStorage();
        SwapHandler swapHandler = $.finalTokenInfos[finalToken].swapHandler;
        if (address(swapHandler) == address(0)) {
            bytes32 salt = _swapHandlerSalt(finalToken);
            swapHandler = new SwapHandler{ salt: salt }();
        }

        $.finalTokenInfos[finalToken] = FinalTokenInfo({
            spotIndex: spotIndex,
            isBuy: isBuy,
            feePpm: feePpm,
            swapHandler: swapHandler,
            suggestedDiscountBps: suggestedDiscountBps
        });

        // We don't allow SwapHandler accounts to be uninitiated. That could lead to loss of funds. They instead should
        // be pre-funded using `predictSwapHandler` to predict their address
        require(HyperCoreLib.coreUserExists(address(swapHandler)), "SwapHandler @ core doesn't exist");

        emit SetFinalTokenInfo(finalToken, spotIndex, isBuy, feePpm, address(swapHandler), suggestedDiscountBps);
    }

    /**
     * @notice Predicts the deterministic address of a SwapHandler for a given finalToken using CREATE2
     * @param finalToken The token address for which to predict the SwapHandler address
     * @return The predicted address of the SwapHandler contract
     */
    function predictSwapHandler(address finalToken) public view returns (address) {
        bytes32 salt = _swapHandlerSalt(finalToken);
        bytes32 initCodeHash = keccak256(type(SwapHandler).creationCode);
        return address(uint160(uint256(keccak256(abi.encodePacked(bytes1(0xff), address(this), salt, initCodeHash)))));
    }

    /// @notice Returns the salt to use when creating a SwapHandler via CREATE2
    function _swapHandlerSalt(address finalToken) internal view returns (bytes32) {
        return keccak256(abi.encodePacked(address(this), finalToken));
    }

    /**************************************
     *            FLOW FUNCTIONS          *
     **************************************/

    /**
     * @notice External entrypoint to execute flow when called via delegatecall from a handler. Works with params
     * checked by a handler. Params authorization by a handler is enforced via `onlyAuthorizedFlow` modifier
     */
    function executeFlow(CommonFlowParams memory params, uint256 maxUserSlippageBps) external onlyAuthorizedFlow {
        if (params.finalToken == baseToken) {
            _executeSimpleTransferFlow(params);
        } else {
            _initiateSwapFlow(params, maxUserSlippageBps);
        }
    }

    /// @notice External entrypoint to execute simple transfer flow (see `executeFlow` comment for details)
    function executeSimpleTransferFlow(CommonFlowParams memory params) external onlyAuthorizedFlow {
        _executeSimpleTransferFlow(params);
    }

    /// @notice External entrypoint to execute fallback evm flow (see `executeFlow` comment for details)
    function fallbackHyperEVMFlow(CommonFlowParams memory params) external onlyAuthorizedFlow {
        _fallbackHyperEVMFlow(params);
    }

    /// @notice Execute a simple transfer flow in which we transfer `finalToken` to the user on HyperCore after receiving
    /// an amount of finalToken from the user on HyperEVM
    function _executeSimpleTransferFlow(CommonFlowParams memory params) internal {
        address finalToken = params.finalToken;
        MainStorage storage $ = _getMainStorage();
        CoreTokenInfo memory coreTokenInfo = $.coreTokenInfos[finalToken];

        uint64 accountActivationFeeCore;
        if (!HyperCoreLib.coreUserExists(params.finalRecipient)) {
            bool isStandard = params.accountCreationMode == AccountCreationMode.Standard;

            // Standard, sponsored
            if (isStandard && params.maxBpsToSponsor > 0) {
                revert AccountNotActivatedError(params.finalRecipient);
            }

            // Standard, non-sponsored OR
            // FromUserAccount AND token can't be used for account activation
            if (isStandard || !coreTokenInfo.canBeUsedForAccountActivation) {
                emit AccountNotActivated(params.quoteNonce, params.finalRecipient);
                _fallbackHyperEVMFlow(params);
                return;
            }

            // FromUserAccount AND token can be used for account activation
            accountActivationFeeCore = coreTokenInfo.accountActivationFeeCore;
        }

        uint256 amountToSponsor;
        {
            uint256 maxEvmAmountToSponsor = ((params.amountInEVM + params.extraFeesIncurred) * params.maxBpsToSponsor) /
                BPS_SCALAR;
            amountToSponsor = params.extraFeesIncurred;
            if (amountToSponsor > maxEvmAmountToSponsor) {
                amountToSponsor = maxEvmAmountToSponsor;
            }

            if (amountToSponsor > 0) {
                if (!_availableInDonationBox(params.quoteNonce, finalToken, amountToSponsor)) {
                    // If the full amount is not available in the donation box, use the balance of the token in the donation box
                    amountToSponsor = IERC20(finalToken).balanceOf(address(donationBox));
                }
            }
        }

        // Calculate quoted amounts and check safety
        uint256 quotedEvmAmount;
        uint64 quotedCoreAmount;
        {
            uint256 finalAmount = params.amountInEVM + amountToSponsor;
            (quotedEvmAmount, quotedCoreAmount) = HyperCoreLib.maximumEVMSendAmountToAmounts(
                finalAmount,
                coreTokenInfo.tokenInfo.evmExtraWeiDecimals
            );
            // If there are no funds left on the destination side of the bridge, the funds will be lost in the
            // bridge. We check send safety via `isCoreAmountSafeToBridge`
            if (
                !HyperCoreLib.isCoreAmountSafeToBridge(
                    coreTokenInfo.coreIndex,
                    quotedCoreAmount,
                    coreTokenInfo.bridgeSafetyBufferCore
                )
            ) {
                // If the amount is not safe to bridge because the bridge doesn't have enough liquidity,
                // fall back to sending user funds on HyperEVM.
                _fallbackHyperEVMFlow(params);
                emit UnsafeToBridge(params.quoteNonce, finalToken, quotedCoreAmount);
                return;
            }
        }

        if (amountToSponsor > 0) {
            // This will succeed because we checked the balance earlier
            donationBox.withdraw(IERC20(finalToken), amountToSponsor);
        }

        $.cumulativeSponsoredAmount[finalToken] += amountToSponsor;

        // There is a very slim change that someone is sending > buffer amount in the same EVM block and the balance of
        // the bridge is not enough to cover our transfer, so the funds are lost.
        HyperCoreLib.transferERC20EVMToCore(
            finalToken,
            coreTokenInfo.coreIndex,
            params.finalRecipient,
            quotedEvmAmount,
            coreTokenInfo.tokenInfo.evmExtraWeiDecimals,
            params.destinationDex,
            accountActivationFeeCore
        );

        if (accountActivationFeeCore > 0) {
            emit AccountActivatedFromUserFunds(
                params.quoteNonce,
                params.finalRecipient,
                finalToken,
                accountActivationFeeCore
            );
        }

        emit SimpleTransferFlowCompleted(
            params.quoteNonce,
            params.finalRecipient,
            finalToken,
            params.amountInEVM,
            params.extraFeesIncurred,
            amountToSponsor
        );
    }

    /**
     * @notice Initiates the swap flow. Sends the funds received on EVM side over to a SwapHandler corresponding to a
     * finalToken. This is the first leg of the swap flow. Next, the bot should submit a limit order through a `submitLimitOrderFromBot`
     * function, and then settle the flow via a `finalizeSwapFlows` function
     * @dev Only works for stable -> stable swap flows (or equivalent token flows. Price between tokens is supposed to be approximately one to one)
     * @param maxUserSlippageBps Describes a configured user setting. Slippage here is wrt the one to one exchange
     */
    function _initiateSwapFlow(CommonFlowParams memory params, uint256 maxUserSlippageBps) internal {
        address initialToken = baseToken;

        MainStorage storage $ = _getMainStorage();
        CoreTokenInfo memory initialCoreTokenInfo = $.coreTokenInfos[initialToken];
        CoreTokenInfo memory finalCoreTokenInfo = $.coreTokenInfos[params.finalToken];
        FinalTokenInfo memory finalTokenInfo = _getExistingFinalTokenInfo(params.finalToken);

        if (!HyperCoreLib.coreUserExists(params.finalRecipient)) {
            bool isStandard = params.accountCreationMode == AccountCreationMode.Standard;

            // Standard, sponsored
            if (isStandard && params.maxBpsToSponsor > 0) {
                revert AccountNotActivatedError(params.finalRecipient);
            }

            // Standard, non-sponsored OR
            // FromUserFunds AND final token can't be used for account creation
            if (isStandard || !finalCoreTokenInfo.canBeUsedForAccountActivation) {
                emit AccountNotActivated(params.quoteNonce, params.finalRecipient);
                params.finalToken = initialToken;
                _fallbackHyperEVMFlow(params);
                return;
            }
        }

        // Calculate limit order amounts and check if feasible
        uint64 minAllowableAmountToForwardCore;
        uint64 maxAllowableAmountToForwardCore;
        // Estimated slippage in ppm, as compared to a one-to-one totalBridgedAmount -> finalAmount conversion
        uint256 estSlippagePpm;
        {
            // In finalToken
            (minAllowableAmountToForwardCore, maxAllowableAmountToForwardCore) = _calcAllowableAmtsSwapFlow(
                params.amountInEVM,
                params.extraFeesIncurred,
                initialCoreTokenInfo,
                finalCoreTokenInfo,
                params.maxBpsToSponsor > 0,
                maxUserSlippageBps
            );

            uint64 approxExecutionPriceX1e8 = _getApproxRealizedPrice(
                finalTokenInfo,
                finalCoreTokenInfo,
                initialCoreTokenInfo
            );
            uint256 maxAllowableBpsDeviation = params.maxBpsToSponsor > 0 ? params.maxBpsToSponsor : maxUserSlippageBps;
            if (finalTokenInfo.isBuy) {
                if (approxExecutionPriceX1e8 < ONEX1e8) {
                    estSlippagePpm = 0;
                } else {
                    // ceil
                    estSlippagePpm = ((approxExecutionPriceX1e8 - ONEX1e8) * PPM_SCALAR + (ONEX1e8 - 1)) / ONEX1e8;
                }
            } else {
                if (approxExecutionPriceX1e8 > ONEX1e8) {
                    estSlippagePpm = 0;
                } else {
                    // ceil
                    estSlippagePpm = ((ONEX1e8 - approxExecutionPriceX1e8) * PPM_SCALAR + (ONEX1e8 - 1)) / ONEX1e8;
                }
            }
            // Add `extraFeesIncurred` to "slippage from one to one"
            estSlippagePpm +=
                (params.extraFeesIncurred * PPM_SCALAR + (params.amountInEVM + params.extraFeesIncurred) - 1) /
                (params.amountInEVM + params.extraFeesIncurred);

            if (estSlippagePpm > maxAllowableBpsDeviation * 10 ** (PPM_DECIMALS - BPS_DECIMALS)) {
                emit SwapFlowTooExpensive(
                    params.quoteNonce,
                    params.finalToken,
                    (estSlippagePpm + 10 ** (PPM_DECIMALS - BPS_DECIMALS) - 1) / 10 ** (PPM_DECIMALS - BPS_DECIMALS),
                    maxAllowableBpsDeviation
                );
                params.finalToken = initialToken;
                _executeSimpleTransferFlow(params);
                return;
            }
        }

        (uint256 tokensToSendEvm, uint64 coreAmountIn) = HyperCoreLib.maximumEVMSendAmountToAmounts(
            params.amountInEVM,
            initialCoreTokenInfo.tokenInfo.evmExtraWeiDecimals
        );

        // Check that we can safely bridge to HCore (for the trade amount actually needed)
        bool isSafeToBridgeMainToken = HyperCoreLib.isCoreAmountSafeToBridge(
            initialCoreTokenInfo.coreIndex,
            coreAmountIn,
            initialCoreTokenInfo.bridgeSafetyBufferCore
        );

        if (!isSafeToBridgeMainToken) {
            emit UnsafeToBridge(params.quoteNonce, initialToken, coreAmountIn);
            params.finalToken = initialToken;
            _fallbackHyperEVMFlow(params);
            return;
        }

        // Finalize swap flow setup by updating state and funding SwapHandler
        // State changes
        $.swaps[params.quoteNonce] = SwapFlowState({
            finalRecipient: params.finalRecipient,
            finalToken: params.finalToken,
            destinationDex: params.destinationDex,
            minAmountToSend: minAllowableAmountToForwardCore,
            maxAmountToSend: maxAllowableAmountToForwardCore,
            isSponsored: params.maxBpsToSponsor > 0,
            finalized: false
        });

        emit SwapFlowInitialized(
            params.quoteNonce,
            params.finalRecipient,
            params.finalToken,
            params.amountInEVM,
            params.extraFeesIncurred,
            coreAmountIn,
            minAllowableAmountToForwardCore,
            maxAllowableAmountToForwardCore
        );

        // Send amount received from user to a corresponding SwapHandler
        SwapHandler swapHandler = finalTokenInfo.swapHandler;
        IERC20(initialToken).safeTransfer(address(swapHandler), tokensToSendEvm);
        swapHandler.transferFundsToSelfOnCore(
            initialToken,
            initialCoreTokenInfo.coreIndex,
            tokensToSendEvm,
            initialCoreTokenInfo.tokenInfo.evmExtraWeiDecimals
        );
    }

    /**
     * @notice Finalizes multiple swap flows associated with a final token, subject to the L1 Hyperliquid balance
     * @dev Caller is responsible for providing correct limitOrderOutput amounts per assosicated swap flow. The caller
     * has to estimate how much final tokens it received on core based on the input of the corresponding quote nonce
     * swap flow
     * @param finalToken The token address for the swaps being finalized
     * @param quoteNonces Array of quote nonces identifying the swap flows to finalize
     * @param limitOrderOuts Array of limit order output amounts corresponding to each quote nonce
     * @return finalized The number of swap flows that were successfully finalized
     */
    function finalizeSwapFlows(
        address finalToken,
        bytes32[] calldata quoteNonces,
        uint64[] calldata limitOrderOuts
    ) external onlyRole(PERMISSIONED_BOT_ROLE) returns (uint256 finalized) {
        MainStorage storage $ = _getMainStorage();
        require(quoteNonces.length == limitOrderOuts.length, "length");
        require($.lastPullFundsBlock[finalToken] < block.number, "too soon");

        CoreTokenInfo memory finalCoreTokenInfo = _getExistingCoreTokenInfo(finalToken);
        FinalTokenInfo memory finalTokenInfo = _getExistingFinalTokenInfo(finalToken);

        uint64 availableBalance = HyperCoreLib.spotBalance(
            address(finalTokenInfo.swapHandler),
            finalCoreTokenInfo.coreIndex
        );
        uint64 totalAdditionalToSend = 0;
        for (; finalized < quoteNonces.length; ++finalized) {
            bool success;
            uint64 additionalToSend;
            (success, additionalToSend, availableBalance) = _finalizeSingleSwap(
                quoteNonces[finalized],
                limitOrderOuts[finalized],
                finalCoreTokenInfo,
                finalTokenInfo.swapHandler,
                finalToken,
                availableBalance
            );
            if (!success) {
                break;
            }
            totalAdditionalToSend += additionalToSend;
        }

        if (finalized > 0) {
            $.lastPullFundsBlock[finalToken] = block.number;
        } else {
            return 0;
        }

        if (totalAdditionalToSend > 0) {
            (uint256 totalAdditionalToSendEVM, uint64 totalAdditionalReceivedCore) = HyperCoreLib
                .minimumCoreReceiveAmountToAmounts(
                    totalAdditionalToSend,
                    finalCoreTokenInfo.tokenInfo.evmExtraWeiDecimals
                );

            if (
                !HyperCoreLib.isCoreAmountSafeToBridge(
                    finalCoreTokenInfo.coreIndex,
                    totalAdditionalReceivedCore,
                    finalCoreTokenInfo.bridgeSafetyBufferCore
                )
            ) {
                // We expect this situation to be so rare and / or intermittend that we're willing to rely on admin to sweep the funds if this leads to
                // swaps being impossible to finalize
                revert UnsafeToBridgeError(finalToken, totalAdditionalToSend);
            }

            $.cumulativeSponsoredAmount[finalToken] += totalAdditionalToSendEVM;

            // ! Notice: as per HyperEVM <> HyperCore rules, this amount will land on HyperCore *before* all of the core > core sends get executed
            // Get additional amount to send from donation box, and send it to self on core
            donationBox.withdraw(IERC20(finalToken), totalAdditionalToSendEVM);
            IERC20(finalToken).safeTransfer(address(finalTokenInfo.swapHandler), totalAdditionalToSendEVM);
            finalTokenInfo.swapHandler.transferFundsToSelfOnCore(
                finalToken,
                finalCoreTokenInfo.coreIndex,
                totalAdditionalToSendEVM,
                finalCoreTokenInfo.tokenInfo.evmExtraWeiDecimals
            );
        }
    }

    /// @notice Finalizes a single swap flow, sending the tokens to user on core. Relies on caller to send the `additionalToSend`
    function _finalizeSingleSwap(
        bytes32 quoteNonce,
        uint64 limitOrderOut,
        CoreTokenInfo memory finalCoreTokenInfo,
        SwapHandler swapHandler,
        address finalToken,
        uint64 availableBalance
    ) internal returns (bool success, uint64 additionalToSend, uint64 balanceRemaining) {
        SwapFlowState storage swap = _getMainStorage().swaps[quoteNonce];
        if (swap.finalRecipient == address(0)) revert SwapDoesNotExist();
        if (swap.finalized) revert SwapAlreadyFinalized();
        if (swap.finalToken != finalToken) revert WrongSwapFinalizationToken(quoteNonce);

        uint64 accountActivationFee;
        // User account can be absent if our AccountCreationMode is `FromUserFunds`. We need to adjust our transfer amount to user based on this
        if (!HyperCoreLib.coreUserExists(swap.finalRecipient)) {
            // This is enforced by `_initiateSwapFlow`. If the setting changed, this revert can trigger. Requires manual resolution (creating an account)
            require(finalCoreTokenInfo.canBeUsedForAccountActivation, TokenNotEligibleForActivation());
            accountActivationFee = finalCoreTokenInfo.accountActivationFeeCore;
        }

        uint64 amountToTransfer;
        uint64 totalConsumed;
        (amountToTransfer, additionalToSend, totalConsumed) = _calcSwapFlowSendAmounts(
            limitOrderOut,
            swap.minAmountToSend,
            swap.maxAmountToSend,
            swap.isSponsored,
            accountActivationFee
        );

        // `additionalToSend` will land on HCore before this core > core send will need to be executed
        balanceRemaining = availableBalance + additionalToSend;
        if (totalConsumed > balanceRemaining) {
            return (false, 0, availableBalance);
        }

        swap.finalized = true;
        success = true;
        balanceRemaining -= totalConsumed;

        (uint256 additionalToSendEVM, ) = HyperCoreLib.minimumCoreReceiveAmountToAmounts(
            additionalToSend,
            finalCoreTokenInfo.tokenInfo.evmExtraWeiDecimals
        );

        swapHandler.transferFundsToUserOnCore(
            finalCoreTokenInfo.coreIndex,
            swap.finalRecipient,
            amountToTransfer,
            swap.destinationDex
        );

        if (accountActivationFee > 0) {
            emit AccountActivatedFromUserFunds(quoteNonce, swap.finalRecipient, swap.finalToken, accountActivationFee);
        }

        emit SwapFlowFinalized(quoteNonce, swap.finalRecipient, swap.finalToken, amountToTransfer, additionalToSendEVM);
    }

    /// @notice Forwards `amount` plus potential sponsorship funds (for bridging fee) to user on HyperEVM
    function _fallbackHyperEVMFlow(CommonFlowParams memory params) internal {
        uint256 maxEvmAmountToSponsor = ((params.amountInEVM + params.extraFeesIncurred) * params.maxBpsToSponsor) /
            BPS_SCALAR;
        uint256 sponsorshipFundsToForward = params.extraFeesIncurred > maxEvmAmountToSponsor
            ? maxEvmAmountToSponsor
            : params.extraFeesIncurred;

        if (!_availableInDonationBox(params.quoteNonce, params.finalToken, sponsorshipFundsToForward)) {
            sponsorshipFundsToForward = 0;
        }
        if (sponsorshipFundsToForward > 0) {
            donationBox.withdraw(IERC20(params.finalToken), sponsorshipFundsToForward);
        }
        uint256 totalAmountToForward = params.amountInEVM + sponsorshipFundsToForward;
        IERC20(params.finalToken).safeTransfer(params.finalRecipient, totalAmountToForward);
        _getMainStorage().cumulativeSponsoredAmount[params.finalToken] += sponsorshipFundsToForward;
        emit FallbackHyperEVMFlowCompleted(
            params.quoteNonce,
            params.finalRecipient,
            params.finalToken,
            params.amountInEVM,
            params.extraFeesIncurred,
            sponsorshipFundsToForward
        );
    }

    /**
     * @notice Activates a user account on Core by funding the account activation fee.
     * @param quoteNonce The nonce of the quote that is used to identify the user.
     * @param finalRecipient The address of the recipient of the funds.
     * @param fundingToken The address of the token that is used to fund the account activation fee.
     */
    function activateUserAccount(
        bytes32 quoteNonce,
        address finalRecipient,
        address fundingToken
    ) external onlyRole(PERMISSIONED_BOT_ROLE) {
        CoreTokenInfo memory coreTokenInfo = _getExistingCoreTokenInfo(fundingToken);
        bool coreUserExists = HyperCoreLib.coreUserExists(finalRecipient);
        require(!coreUserExists, "Can't fund account activation for existing user");
        require(coreTokenInfo.canBeUsedForAccountActivation, "Token can't be used for this");

        // +1 wei for a spot send
        uint64 totalBalanceRequiredToActivate = coreTokenInfo.accountActivationFeeCore + 1;
        (uint256 evmAmountToSend, ) = HyperCoreLib.minimumCoreReceiveAmountToAmounts(
            totalBalanceRequiredToActivate,
            coreTokenInfo.tokenInfo.evmExtraWeiDecimals
        );

        bool safeToBridge = HyperCoreLib.isCoreAmountSafeToBridge(
            coreTokenInfo.coreIndex,
            totalBalanceRequiredToActivate,
            coreTokenInfo.bridgeSafetyBufferCore
        );
        require(safeToBridge, "Not safe to bridge");
        _getMainStorage().cumulativeSponsoredActivationFee[fundingToken] += evmAmountToSend;

        // donationBox @ evm -> Handler @ evm
        donationBox.withdraw(IERC20(fundingToken), evmAmountToSend);

        HyperCoreLib.activateCoreAccountFromEVM(fundingToken, coreTokenInfo.coreIndex, finalRecipient, evmAmountToSend);

        emit SponsoredAccountActivation(quoteNonce, finalRecipient, fundingToken, evmAmountToSend);
    }

    /**
     * @notice Cancells a pending limit order by `cloid` with an intention to submit a new limit order in its place. To
     * be used for stale limit orders to speed up executing user transactions
     * @param finalToken The token address for which the limit order was placed
     * @param cloid Client order ID of the limit order to cancel
     */
    function cancelLimitOrderByCloid(address finalToken, uint128 cloid) external onlyRole(PERMISSIONED_BOT_ROLE) {
        FinalTokenInfo memory finalTokenInfo = _getExistingFinalTokenInfo(finalToken);
        finalTokenInfo.swapHandler.cancelOrderByCloid(finalTokenInfo.spotIndex, cloid);

        emit CancelledLimitOrder(finalToken, cloid);
    }

    function submitLimitOrderFromBot(
        address finalToken,
        uint64 priceX1e8,
        uint64 sizeX1e8,
        uint128 cloid
    ) external onlyRole(PERMISSIONED_BOT_ROLE) {
        FinalTokenInfo memory finalTokenInfo = _getExistingFinalTokenInfo(finalToken);
        finalTokenInfo.swapHandler.submitSpotLimitOrder(finalTokenInfo, priceX1e8, sizeX1e8, cloid);

        emit SubmittedLimitOrder(finalToken, priceX1e8, sizeX1e8, cloid);
    }

    function _setCoreTokenInfo(
        address token,
        uint32 coreIndex,
        bool canBeUsedForAccountActivation,
        uint64 accountActivationFeeCore,
        uint64 bridgeSafetyBufferCore
    ) internal {
        _getMainStorage().coreTokenInfos[token] = HyperCoreLib.buildCoreTokenInfo(
            coreIndex,
            canBeUsedForAccountActivation,
            accountActivationFeeCore,
            bridgeSafetyBufferCore
        );

        emit SetCoreTokenInfo(
            token,
            coreIndex,
            canBeUsedForAccountActivation,
            accountActivationFeeCore,
            bridgeSafetyBufferCore
        );
    }

    /**
     * @notice Used for ad-hoc sends of sponsorship funds to associated SwapHandler @ HyperCore
     * @param token The final token for which we want to fund the SwapHandler
     * @param amount The amount of tokens to send to the SwapHandler
     */
    function sendSponsorshipFundsToSwapHandler(address token, uint256 amount) external onlyRole(PERMISSIONED_BOT_ROLE) {
        CoreTokenInfo memory coreTokenInfo = _getExistingCoreTokenInfo(token);
        FinalTokenInfo memory finalTokenInfo = _getExistingFinalTokenInfo(token);
        (uint256 amountEVMToSend, uint64 amountCoreToReceive) = HyperCoreLib.maximumEVMSendAmountToAmounts(
            amount,
            coreTokenInfo.tokenInfo.evmExtraWeiDecimals
        );
        if (
            !HyperCoreLib.isCoreAmountSafeToBridge(
                coreTokenInfo.coreIndex,
                amountCoreToReceive,
                coreTokenInfo.bridgeSafetyBufferCore
            )
        ) {
            revert UnsafeToBridgeError(token, amountCoreToReceive);
        }

        _getMainStorage().cumulativeSponsoredAmount[token] += amountEVMToSend;

        emit SentSponsorshipFundsToSwapHandler(token, amountEVMToSend);

        donationBox.withdraw(IERC20(token), amountEVMToSend);
        IERC20(token).safeTransfer(address(finalTokenInfo.swapHandler), amountEVMToSend);
        finalTokenInfo.swapHandler.transferFundsToSelfOnCore(
            token,
            coreTokenInfo.coreIndex,
            amountEVMToSend,
            coreTokenInfo.tokenInfo.evmExtraWeiDecimals
        );
    }

    /// @notice Checks if `amount` of `token` is available to withdraw from donationBox
    function _availableInDonationBox(
        bytes32 quoteNonce,
        address token,
        uint256 amount
    ) internal returns (bool available) {
        uint256 balance = IERC20(token).balanceOf(address(donationBox));
        available = balance >= amount;
        if (!available) {
            emit DonationBoxInsufficientFunds(quoteNonce, token, amount, balance);
        }
    }

    function _calcAllowableAmtsSwapFlow(
        uint256 amount,
        uint256 extraFeesIncurred,
        CoreTokenInfo memory initialCoreTokenInfo,
        CoreTokenInfo memory finalCoreTokenInfo,
        bool isSponsoredFlow,
        uint256 maxUserSlippageBps
    ) internal pure returns (uint64 minAllowableAmountToForwardCore, uint64 maxAllowableAmountToForwardCore) {
        (, uint64 feelessAmountCoreInitialToken) = HyperCoreLib.maximumEVMSendAmountToAmounts(
            amount + extraFeesIncurred,
            initialCoreTokenInfo.tokenInfo.evmExtraWeiDecimals
        );
        uint64 feelessAmountCoreFinalToken = HyperCoreLib.convertCoreDecimalsSimple(
            feelessAmountCoreInitialToken,
            initialCoreTokenInfo.tokenInfo.weiDecimals,
            finalCoreTokenInfo.tokenInfo.weiDecimals
        );
        if (isSponsoredFlow) {
            minAllowableAmountToForwardCore = feelessAmountCoreFinalToken;
            maxAllowableAmountToForwardCore = feelessAmountCoreFinalToken;
        } else {
            minAllowableAmountToForwardCore = uint64(
                (feelessAmountCoreFinalToken * (BPS_SCALAR - maxUserSlippageBps)) / BPS_SCALAR
            );
            maxAllowableAmountToForwardCore = feelessAmountCoreFinalToken;
        }
    }

    /**
     * @return amountToTransfer What we will forward to user on HCore
     * @return additionalToSend What we will send from donationBox right now
     * @return totalAttributableToUser Total amount that we reserve for this user. The amount consumed from SwapHandler
                                       during send execution. Differs from `amountToTransfer` if account creation fee is
                                       present
     */
    function _calcSwapFlowSendAmounts(
        uint64 limitOrderOut,
        uint64 minAmountToSend,
        uint64 maxAmountToSend,
        bool isSponsored,
        uint64 accountActivationFee
    ) internal pure returns (uint64 amountToTransfer, uint64 additionalToSend, uint64 totalAttributableToUser) {
        if (isSponsored) {
            // `minAmountToSend` is equal to `maxAmountToSend` for the sponsored flow
            totalAttributableToUser = minAmountToSend;
            additionalToSend = totalAttributableToUser > limitOrderOut ? totalAttributableToUser - limitOrderOut : 0;
        } else {
            additionalToSend = limitOrderOut < minAmountToSend ? minAmountToSend - limitOrderOut : 0;
            uint64 proposedToSend = limitOrderOut + additionalToSend;
            totalAttributableToUser = proposedToSend > maxAmountToSend ? maxAmountToSend : proposedToSend;
        }

        // If the order total is not enough to cover account creation, the account has to be created separately (by the User or Bot) to be able to finalize this swap flow
        require(totalAttributableToUser > accountActivationFee, InsufficientFundsForActivation());

        amountToTransfer = totalAttributableToUser - accountActivationFee;
    }

    /// @notice Reads the current spot price from HyperLiquid and applies a configured suggested discount for faster execution
    /// @dev Includes HyperLiquid fees
    function _getApproxRealizedPrice(
        FinalTokenInfo memory finalTokenInfo,
        CoreTokenInfo memory finalCoreTokenInfo,
        CoreTokenInfo memory initialCoreTokenInfo
    ) internal view returns (uint64 limitPriceX1e8) {
        uint256 spotPxRaw = HyperCoreLib.spotPx(finalTokenInfo.spotIndex);
        // Convert to 10 ** 8 precision (https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/hyperevm/interacting-with-hypercore)
        // `szDecimals` of the base aseet for spot market
        uint8 additionalPowersOf10 = finalTokenInfo.isBuy
            ? finalCoreTokenInfo.tokenInfo.szDecimals
            : initialCoreTokenInfo.tokenInfo.szDecimals;
        uint256 spotX1e8 = spotPxRaw * (10 ** additionalPowersOf10);

        // Buy above spot, sell below spot
        uint256 adjPpm = finalTokenInfo.isBuy
            ? (PPM_SCALAR + finalTokenInfo.suggestedDiscountBps * 10 ** 2 + finalTokenInfo.feePpm)
            : (PPM_SCALAR - finalTokenInfo.suggestedDiscountBps * 10 ** 2 - finalTokenInfo.feePpm);
        limitPriceX1e8 = uint64((uint256(spotX1e8) * adjPpm) / PPM_SCALAR);
    }

    /**************************************
     *            SWEEP FUNCTIONS         *
     **************************************/

    function sweepNative(uint256 amount) external onlyRole(FUNDS_SWEEPER_ROLE) {
        (bool success, ) = msg.sender.call{ value: amount }("");
        require(success, "Failed to send native funds");
    }

    function sweepErc20(address token, uint256 amount) external onlyRole(FUNDS_SWEEPER_ROLE) {
        IERC20(token).safeTransfer(msg.sender, amount);
    }

    function sweepErc20FromDonationBox(address token, uint256 amount) external onlyRole(FUNDS_SWEEPER_ROLE) {
        donationBox.withdraw(IERC20(token), amount);
        IERC20(token).safeTransfer(msg.sender, amount);
    }

    function sweepERC20FromSwapHandler(address token, uint256 amount) external onlyRole(FUNDS_SWEEPER_ROLE) {
        SwapHandler swapHandler = _getExistingFinalTokenInfo(token).swapHandler;
        swapHandler.sweepErc20(token, amount);
        IERC20(token).safeTransfer(msg.sender, amount);
    }

    function sweepOnCore(
        address token,
        uint64 amount,
        uint32 sourceDex,
        uint32 destinationDex
    ) external onlyRole(FUNDS_SWEEPER_ROLE) {
        HyperCoreLib.transferERC20CoreToCore(
            _getMainStorage().coreTokenInfos[token].coreIndex,
            msg.sender,
            amount,
            sourceDex,
            destinationDex
        );
    }

    function sweepOnCoreFromSwapHandler(
        address finalToken,
        uint64 finalTokenAmount,
        uint64 baseTokenAmount
    ) external onlyRole(FUNDS_SWEEPER_ROLE) {
        MainStorage storage $ = _getMainStorage();
        require($.lastPullFundsBlock[finalToken] < block.number, "Can't pull funds twice in the same block");
        $.lastPullFundsBlock[finalToken] = block.number;

        SwapHandler swapHandler = $.finalTokenInfos[finalToken].swapHandler;

        // funds in swap handler will always be in spot dex
        if (finalTokenAmount > 0) {
            swapHandler.transferFundsToUserOnCore(
                $.coreTokenInfos[finalToken].coreIndex,
                msg.sender,
                finalTokenAmount,
                HyperCoreLib.CORE_SPOT_DEX_ID
            );
        }
        if (baseTokenAmount > 0) {
            swapHandler.transferFundsToUserOnCore(
                $.coreTokenInfos[baseToken].coreIndex,
                msg.sender,
                baseTokenAmount,
                HyperCoreLib.CORE_SPOT_DEX_ID
            );
        }
    }
}


// ===== FILE: contracts/periphery/mintburn/ArbitraryEVMFlowExecutor.sol =====
// SPDX-License-Identifier: BUSL-1.1
pragma solidity ^0.8.0;

import { IERC20 } from "@openzeppelin/contracts-v4/token/ERC20/IERC20.sol";
import { SafeERC20 } from "@openzeppelin/contracts-v4/token/ERC20/utils/SafeERC20.sol";

// Import MulticallHandler
import { MulticallHandler } from "../../handlers/MulticallHandler.sol";
import { EVMFlowParams, CommonFlowParams } from "./Structs.sol";

/**
 * @title ArbitraryEVMFlowExecutor
 * @notice Base contract for executing arbitrary action sequences using MulticallHandler
 * @dev This contract provides shared functionality for both OFT and CCTP handlers to execute
 * arbitrary actions on HyperEVM via MulticallHandler, returning information about the resulting token amount
 * @custom:security-contact bugs@across.to
 */
abstract contract ArbitraryEVMFlowExecutor {
    using SafeERC20 for IERC20;

    /// @notice Compressed call struct (no value field to save gas)
    struct CompressedCall {
        address target;
        bytes callData;
    }

    /// @notice MulticallHandler contract instance
    address public immutable multicallHandler;

    /**
     * @notice Emitted when arbitrary actions are executed successfully
     * @param quoteNonce Unique identifier for this quote/transaction
     * @param initialToken The token address received before executing actions
     * @param initialAmount The amount of initial token received
     * @param finalToken The token address after executing actions
     * @param finalAmount The amount of final token after executing actions
     */
    event ArbitraryActionsExecuted(
        bytes32 indexed quoteNonce,
        address indexed initialToken,
        uint256 initialAmount,
        address indexed finalToken,
        uint256 finalAmount
    );

    uint256 private constant BPS_TOTAL_PRECISION = 18;
    uint256 private constant BPS_PRECISION_SCALAR = 10 ** BPS_TOTAL_PRECISION;

    constructor(address _multicallHandler) {
        multicallHandler = _multicallHandler;
    }

    /**
     * @notice Executes arbitrary actions by transferring tokens to MulticallHandler
     * @dev Decompresses CompressedCall[] to MulticallHandler.Call[] format (adds value: 0)
     * @param params Parameters of HyperEVM execution
     * @return commonParams Parameters to continue sponsored execution to transfer funds to final recipient at correct destination
     */
    function _executeFlow(EVMFlowParams memory params) internal returns (CommonFlowParams memory commonParams) {
        // Decode the compressed action data
        CompressedCall[] memory compressedCalls = abi.decode(params.actionData, (CompressedCall[]));

        // Sweep any pre-existing dust on MulticallHandler so it cannot pollute our balance snapshots
        _drainMulticallHandlerDust(params.initialToken);
        if (params.commonParams.finalToken != params.initialToken) {
            _drainMulticallHandlerDust(params.commonParams.finalToken);
        }

        bool differentTokens = params.initialToken != params.commonParams.finalToken;

        // Read "starting balance initial token"(sBI) and "starting balance final token"(sBF)
        uint256 sBI = IERC20(params.initialToken).balanceOf(address(this));
        uint256 sBF = differentTokens ? IERC20(params.commonParams.finalToken).balanceOf(address(this)) : sBI;

        // Transfer tokens to MulticallHandler
        IERC20(params.initialToken).safeTransfer(multicallHandler, params.commonParams.amountInEVM);

        // Build instructions for MulticallHandler
        bytes memory instructions = _buildMulticallInstructions(
            compressedCalls,
            params.commonParams.finalToken,
            address(this) // Send leftover tokens back to this contract
        );

        // Execute via MulticallHandler
        MulticallHandler(payable(multicallHandler)).handleV3AcrossMessage(
            params.initialToken,
            params.commonParams.amountInEVM,
            address(this),
            instructions
        );

        // Default to initial-token accounting; overwrite below if finalToken was actually produced.
        // Ending balance initial token
        uint256 eBI = IERC20(params.initialToken).balanceOf(address(this));
        uint256 finalAmount = params.commonParams.amountInEVM + eBI - sBI;
        if (differentTokens) {
            // Ending balance final token
            uint256 eBF = IERC20(params.commonParams.finalToken).balanceOf(address(this));

            // Any positive finalToken delta is treated as successful execution when initialToken != finalToken. If any
            // (or all) of the initialToken amount is unspent during the execution, it lands back into this contract and
            // can be withdrawn by the admin
            if (eBF > sBF) {
                finalAmount = eBF - sBF;
            } else {
                params.commonParams.finalToken = params.initialToken;
            }
        }

        params.commonParams.extraFeesIncurred = _calcExtraFeesFinal(
            params.commonParams.amountInEVM,
            params.commonParams.extraFeesIncurred,
            finalAmount
        );
        params.commonParams.amountInEVM = finalAmount;

        emit ArbitraryActionsExecuted(
            params.commonParams.quoteNonce,
            params.initialToken,
            params.commonParams.amountInEVM,
            params.commonParams.finalToken,
            finalAmount
        );

        return params.commonParams;
    }

    /**
     * @notice Builds MulticallHandler Instructions from compressed calls
     * @dev Decompresses calls by adding value: 0, and adds drainLeftoverTokens call at the end
     */
    function _buildMulticallInstructions(
        CompressedCall[] memory compressedCalls,
        address finalToken,
        address fallbackRecipient
    ) internal view returns (bytes memory) {
        uint256 callCount = compressedCalls.length;

        // Create Call[] array with value: 0 for each call, plus one for drainLeftoverTokens
        MulticallHandler.Call[] memory calls = new MulticallHandler.Call[](callCount + 1);

        // Decompress: add value: 0 to each call
        for (uint256 i = 0; i < callCount; ++i) {
            calls[i] = MulticallHandler.Call({
                target: compressedCalls[i].target,
                callData: compressedCalls[i].callData,
                value: 0
            });
        }

        // Add final call to drain leftover tokens back to this contract
        calls[callCount] = MulticallHandler.Call({
            target: multicallHandler,
            callData: abi.encodeCall(MulticallHandler.drainLeftoverTokens, (finalToken, payable(fallbackRecipient))),
            value: 0
        });

        // Build Instructions struct
        MulticallHandler.Instructions memory instructions = MulticallHandler.Instructions({
            calls: calls,
            fallbackRecipient: fallbackRecipient
        });

        return abi.encode(instructions);
    }

    /// @notice Drains any pre-existing balance of token from MulticallHandler
    function _drainMulticallHandlerDust(address token) internal {
        if (IERC20(token).balanceOf(multicallHandler) == 0) return;

        // Empty instructions with `fallbackRecipient` set. Causes MulticallHandler to call `_drainRemainingTokens`, which
        // does a safeTransfer of `token` to `fallbackRecipient`, this contract, essentially removing dust
        bytes memory instructions = abi.encode(
            MulticallHandler.Instructions({ calls: new MulticallHandler.Call[](0), fallbackRecipient: address(this) })
        );

        MulticallHandler(payable(multicallHandler)).handleV3AcrossMessage(token, 0, address(this), instructions);
    }

    /// @notice Calculates proportional fees to sponsor in finalToken, given the fees to sponsor in initial token and initial amount
    function _calcExtraFeesFinal(
        uint256 amount,
        uint256 extraFeesToSponsorTokenIn,
        uint256 finalAmount
    ) internal pure returns (uint256 extraFeesToSponsorFinalToken) {
        // Total amount to sponsor is the extra fees to sponsor, ceiling division.
        uint256 bpsToSponsor;
        {
            uint256 totalAmount = amount + extraFeesToSponsorTokenIn;
            bpsToSponsor = ((extraFeesToSponsorTokenIn * BPS_PRECISION_SCALAR) + totalAmount - 1) / totalAmount;
        }

        // Apply the bps to sponsor to the final amount to get the amount to sponsor, ceiling division.
        uint256 bpsToSponsorAdjusted = BPS_PRECISION_SCALAR - bpsToSponsor;
        extraFeesToSponsorFinalToken =
            (((finalAmount * BPS_PRECISION_SCALAR) + bpsToSponsorAdjusted - 1) / bpsToSponsorAdjusted) - finalAmount;
    }

    /// @notice Allow contract to receive native tokens for arbitrary action execution
    receive() external payable virtual {}
}


// ===== FILE: contracts/periphery/mintburn/Structs.sol =====
// SPDX-License-Identifier: BUSL-1.1
pragma solidity ^0.8.0;

import { SwapHandler } from "./SwapHandler.sol";

enum AccountCreationMode {
    Standard,
    FromUserFunds
}

struct FinalTokenInfo {
    // The index of the market where we're going to swap baseToken -> finalToken
    uint32 spotIndex;
    // To go baseToken -> finalToken, do we have to enqueue a buy or a sell?
    bool isBuy;
    // The fee Hyperliquid charges for Limit orders in the market; in parts per million, e.g. 1.4 bps = 140 ppm
    uint32 feePpm;
    // When enqueuing a limit order, use this to set a price "a bit worse than market" for faster execution
    uint32 suggestedDiscountBps;
    // Contract where the accounting for all baseToken -> finalToken accounting happens. One pre finalToken
    SwapHandler swapHandler;
}

/// @notice Common parameters shared across flow execution functions
struct CommonFlowParams {
    uint256 amountInEVM;
    bytes32 quoteNonce;
    address finalRecipient;
    address finalToken;
    uint32 destinationDex;
    AccountCreationMode accountCreationMode;
    uint256 maxBpsToSponsor;
    uint256 extraFeesIncurred;
}

/// @notice Parameters for executing flows with arbitrary EVM actions
struct EVMFlowParams {
    CommonFlowParams commonParams;
    address initialToken;
    bytes actionData;
    bool transferToCore;
}


// ===== FILE: node_modules/_openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol =====
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


// ===== FILE: node_modules/_openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.5.0) (token/ERC20/utils/SafeERC20.sol)

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
        if (!_safeTransfer(token, to, value, true)) {
            revert SafeERC20FailedOperation(address(token));
        }
    }

    /**
     * @dev Transfer `value` amount of `token` from `from` to `to`, spending the approval given by `from` to the
     * calling contract. If `token` returns no value, non-reverting calls are assumed to be successful.
     */
    function safeTransferFrom(IERC20 token, address from, address to, uint256 value) internal {
        if (!_safeTransferFrom(token, from, to, value, true)) {
            revert SafeERC20FailedOperation(address(token));
        }
    }

    /**
     * @dev Variant of {safeTransfer} that returns a bool instead of reverting if the operation is not successful.
     */
    function trySafeTransfer(IERC20 token, address to, uint256 value) internal returns (bool) {
        return _safeTransfer(token, to, value, false);
    }

    /**
     * @dev Variant of {safeTransferFrom} that returns a bool instead of reverting if the operation is not successful.
     */
    function trySafeTransferFrom(IERC20 token, address from, address to, uint256 value) internal returns (bool) {
        return _safeTransferFrom(token, from, to, value, false);
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
        if (!_safeApprove(token, spender, value, false)) {
            if (!_safeApprove(token, spender, 0, true)) revert SafeERC20FailedOperation(address(token));
            if (!_safeApprove(token, spender, value, true)) revert SafeERC20FailedOperation(address(token));
        }
    }

    /**
     * @dev Performs an {ERC1363} transferAndCall, with a fallback to the simple {ERC20} transfer if the target has no
     * code. This can be used to implement an {ERC721}-like safe transfer that relies on {ERC1363} checks when
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
     * has no code. This can be used to implement an {ERC721}-like safe transfer that relies on {ERC1363} checks when
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
     * Oppositely, when the recipient address (`to`) has code, this function only attempts to call {ERC1363-approveAndCall}
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
     * @dev Imitates a Solidity `token.transfer(to, value)` call, relaxing the requirement on the return value: the
     * return value is optional (but if data is returned, it must not be false).
     *
     * @param token The token targeted by the call.
     * @param to The recipient of the tokens
     * @param value The amount of token to transfer
     * @param bubble Behavior switch if the transfer call reverts: bubble the revert reason or return a false boolean.
     */
    function _safeTransfer(IERC20 token, address to, uint256 value, bool bubble) private returns (bool success) {
        bytes4 selector = IERC20.transfer.selector;

        assembly ("memory-safe") {
            let fmp := mload(0x40)
            mstore(0x00, selector)
            mstore(0x04, and(to, shr(96, not(0))))
            mstore(0x24, value)
            success := call(gas(), token, 0, 0x00, 0x44, 0x00, 0x20)
            // if call success and return is true, all is good.
            // otherwise (not success or return is not true), we need to perform further checks
            if iszero(and(success, eq(mload(0x00), 1))) {
                // if the call was a failure and bubble is enabled, bubble the error
                if and(iszero(success), bubble) {
                    returndatacopy(fmp, 0x00, returndatasize())
                    revert(fmp, returndatasize())
                }
                // if the return value is not true, then the call is only successful if:
                // - the token address has code
                // - the returndata is empty
                success := and(success, and(iszero(returndatasize()), gt(extcodesize(token), 0)))
            }
            mstore(0x40, fmp)
        }
    }

    /**
     * @dev Imitates a Solidity `token.transferFrom(from, to, value)` call, relaxing the requirement on the return
     * value: the return value is optional (but if data is returned, it must not be false).
     *
     * @param token The token targeted by the call.
     * @param from The sender of the tokens
     * @param to The recipient of the tokens
     * @param value The amount of token to transfer
     * @param bubble Behavior switch if the transfer call reverts: bubble the revert reason or return a false boolean.
     */
    function _safeTransferFrom(
        IERC20 token,
        address from,
        address to,
        uint256 value,
        bool bubble
    ) private returns (bool success) {
        bytes4 selector = IERC20.transferFrom.selector;

        assembly ("memory-safe") {
            let fmp := mload(0x40)
            mstore(0x00, selector)
            mstore(0x04, and(from, shr(96, not(0))))
            mstore(0x24, and(to, shr(96, not(0))))
            mstore(0x44, value)
            success := call(gas(), token, 0, 0x00, 0x64, 0x00, 0x20)
            // if call success and return is true, all is good.
            // otherwise (not success or return is not true), we need to perform further checks
            if iszero(and(success, eq(mload(0x00), 1))) {
                // if the call was a failure and bubble is enabled, bubble the error
                if and(iszero(success), bubble) {
                    returndatacopy(fmp, 0x00, returndatasize())
                    revert(fmp, returndatasize())
                }
                // if the return value is not true, then the call is only successful if:
                // - the token address has code
                // - the returndata is empty
                success := and(success, and(iszero(returndatasize()), gt(extcodesize(token), 0)))
            }
            mstore(0x40, fmp)
            mstore(0x60, 0)
        }
    }

    /**
     * @dev Imitates a Solidity `token.approve(spender, value)` call, relaxing the requirement on the return value:
     * the return value is optional (but if data is returned, it must not be false).
     *
     * @param token The token targeted by the call.
     * @param spender The spender of the tokens
     * @param value The amount of token to transfer
     * @param bubble Behavior switch if the transfer call reverts: bubble the revert reason or return a false boolean.
     */
    function _safeApprove(IERC20 token, address spender, uint256 value, bool bubble) private returns (bool success) {
        bytes4 selector = IERC20.approve.selector;

        assembly ("memory-safe") {
            let fmp := mload(0x40)
            mstore(0x00, selector)
            mstore(0x04, and(spender, shr(96, not(0))))
            mstore(0x24, value)
            success := call(gas(), token, 0, 0x00, 0x44, 0x00, 0x20)
            // if call success and return is true, all is good.
            // otherwise (not success or return is not true), we need to perform further checks
            if iszero(and(success, eq(mload(0x00), 1))) {
                // if the call was a failure and bubble is enabled, bubble the error
                if and(iszero(success), bubble) {
                    returndatacopy(fmp, 0x00, returndatasize())
                    revert(fmp, returndatasize())
                }
                // if the return value is not true, then the call is only successful if:
                // - the token address has code
                // - the returndata is empty
                success := and(success, and(iszero(returndatasize()), gt(extcodesize(token), 0)))
            }
            mstore(0x40, fmp)
        }
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts-v4/utils/cryptography/ECDSA.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.0) (utils/cryptography/ECDSA.sol)

pragma solidity ^0.8.0;

import "../Strings.sol";

/**
 * @dev Elliptic Curve Digital Signature Algorithm (ECDSA) operations.
 *
 * These functions can be used to verify that a message was signed by the holder
 * of the private keys of a given address.
 */
library ECDSA {
    enum RecoverError {
        NoError,
        InvalidSignature,
        InvalidSignatureLength,
        InvalidSignatureS,
        InvalidSignatureV // Deprecated in v4.8
    }

    function _throwError(RecoverError error) private pure {
        if (error == RecoverError.NoError) {
            return; // no error: do nothing
        } else if (error == RecoverError.InvalidSignature) {
            revert("ECDSA: invalid signature");
        } else if (error == RecoverError.InvalidSignatureLength) {
            revert("ECDSA: invalid signature length");
        } else if (error == RecoverError.InvalidSignatureS) {
            revert("ECDSA: invalid signature 's' value");
        }
    }

    /**
     * @dev Returns the address that signed a hashed message (`hash`) with
     * `signature` or error string. This address can then be used for verification purposes.
     *
     * The `ecrecover` EVM opcode allows for malleable (non-unique) signatures:
     * this function rejects them by requiring the `s` value to be in the lower
     * half order, and the `v` value to be either 27 or 28.
     *
     * IMPORTANT: `hash` _must_ be the result of a hash operation for the
     * verification to be secure: it is possible to craft signatures that
     * recover to arbitrary addresses for non-hashed data. A safe way to ensure
     * this is by receiving a hash of the original message (which may otherwise
     * be too long), and then calling {toEthSignedMessageHash} on it.
     *
     * Documentation for signature generation:
     * - with https://web3js.readthedocs.io/en/v1.3.4/web3-eth-accounts.html#sign[Web3.js]
     * - with https://docs.ethers.io/v5/api/signer/#Signer-signMessage[ethers]
     *
     * _Available since v4.3._
     */
    function tryRecover(bytes32 hash, bytes memory signature) internal pure returns (address, RecoverError) {
        if (signature.length == 65) {
            bytes32 r;
            bytes32 s;
            uint8 v;
            // ecrecover takes the signature parameters, and the only way to get them
            // currently is to use assembly.
            /// @solidity memory-safe-assembly
            assembly {
                r := mload(add(signature, 0x20))
                s := mload(add(signature, 0x40))
                v := byte(0, mload(add(signature, 0x60)))
            }
            return tryRecover(hash, v, r, s);
        } else {
            return (address(0), RecoverError.InvalidSignatureLength);
        }
    }

    /**
     * @dev Returns the address that signed a hashed message (`hash`) with
     * `signature`. This address can then be used for verification purposes.
     *
     * The `ecrecover` EVM opcode allows for malleable (non-unique) signatures:
     * this function rejects them by requiring the `s` value to be in the lower
     * half order, and the `v` value to be either 27 or 28.
     *
     * IMPORTANT: `hash` _must_ be the result of a hash operation for the
     * verification to be secure: it is possible to craft signatures that
     * recover to arbitrary addresses for non-hashed data. A safe way to ensure
     * this is by receiving a hash of the original message (which may otherwise
     * be too long), and then calling {toEthSignedMessageHash} on it.
     */
    function recover(bytes32 hash, bytes memory signature) internal pure returns (address) {
        (address recovered, RecoverError error) = tryRecover(hash, signature);
        _throwError(error);
        return recovered;
    }

    /**
     * @dev Overload of {ECDSA-tryRecover} that receives the `r` and `vs` short-signature fields separately.
     *
     * See https://eips.ethereum.org/EIPS/eip-2098[EIP-2098 short signatures]
     *
     * _Available since v4.3._
     */
    function tryRecover(bytes32 hash, bytes32 r, bytes32 vs) internal pure returns (address, RecoverError) {
        bytes32 s = vs & bytes32(0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff);
        uint8 v = uint8((uint256(vs) >> 255) + 27);
        return tryRecover(hash, v, r, s);
    }

    /**
     * @dev Overload of {ECDSA-recover} that receives the `r and `vs` short-signature fields separately.
     *
     * _Available since v4.2._
     */
    function recover(bytes32 hash, bytes32 r, bytes32 vs) internal pure returns (address) {
        (address recovered, RecoverError error) = tryRecover(hash, r, vs);
        _throwError(error);
        return recovered;
    }

    /**
     * @dev Overload of {ECDSA-tryRecover} that receives the `v`,
     * `r` and `s` signature fields separately.
     *
     * _Available since v4.3._
     */
    function tryRecover(bytes32 hash, uint8 v, bytes32 r, bytes32 s) internal pure returns (address, RecoverError) {
        // EIP-2 still allows signature malleability for ecrecover(). Remove this possibility and make the signature
        // unique. Appendix F in the Ethereum Yellow paper (https://ethereum.github.io/yellowpaper/paper.pdf), defines
        // the valid range for s in (301): 0 < s < secp256k1n ÷ 2 + 1, and for v in (302): v ∈ {27, 28}. Most
        // signatures from current libraries generate a unique signature with an s-value in the lower half order.
        //
        // If your library generates malleable signatures, such as s-values in the upper range, calculate a new s-value
        // with 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141 - s1 and flip v from 27 to 28 or
        // vice versa. If your library also generates signatures with 0/1 for v instead 27/28, add 27 to v to accept
        // these malleable signatures as well.
        if (uint256(s) > 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0) {
            return (address(0), RecoverError.InvalidSignatureS);
        }

        // If the signature is valid (and not malleable), return the signer address
        address signer = ecrecover(hash, v, r, s);
        if (signer == address(0)) {
            return (address(0), RecoverError.InvalidSignature);
        }

        return (signer, RecoverError.NoError);
    }

    /**
     * @dev Overload of {ECDSA-recover} that receives the `v`,
     * `r` and `s` signature fields separately.
     */
    function recover(bytes32 hash, uint8 v, bytes32 r, bytes32 s) internal pure returns (address) {
        (address recovered, RecoverError error) = tryRecover(hash, v, r, s);
        _throwError(error);
        return recovered;
    }

    /**
     * @dev Returns an Ethereum Signed Message, created from a `hash`. This
     * produces hash corresponding to the one signed with the
     * https://eth.wiki/json-rpc/API#eth_sign[`eth_sign`]
     * JSON-RPC method as part of EIP-191.
     *
     * See {recover}.
     */
    function toEthSignedMessageHash(bytes32 hash) internal pure returns (bytes32 message) {
        // 32 is the length in bytes of hash,
        // enforced by the type signature above
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, "\x19Ethereum Signed Message:\n32")
            mstore(0x1c, hash)
            message := keccak256(0x00, 0x3c)
        }
    }

    /**
     * @dev Returns an Ethereum Signed Message, created from `s`. This
     * produces hash corresponding to the one signed with the
     * https://eth.wiki/json-rpc/API#eth_sign[`eth_sign`]
     * JSON-RPC method as part of EIP-191.
     *
     * See {recover}.
     */
    function toEthSignedMessageHash(bytes memory s) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n", Strings.toString(s.length), s));
    }

    /**
     * @dev Returns an Ethereum Signed Typed Data, created from a
     * `domainSeparator` and a `structHash`. This produces hash corresponding
     * to the one signed with the
     * https://eips.ethereum.org/EIPS/eip-712[`eth_signTypedData`]
     * JSON-RPC method as part of EIP-712.
     *
     * See {recover}.
     */
    function toTypedDataHash(bytes32 domainSeparator, bytes32 structHash) internal pure returns (bytes32 data) {
        /// @solidity memory-safe-assembly
        assembly {
            let ptr := mload(0x40)
            mstore(ptr, "\x19\x01")
            mstore(add(ptr, 0x02), domainSeparator)
            mstore(add(ptr, 0x22), structHash)
            data := keccak256(ptr, 0x42)
        }
    }

    /**
     * @dev Returns an Ethereum Signed Data with intended validator, created from a
     * `validator` and `data` according to the version 0 of EIP-191.
     *
     * See {recover}.
     */
    function toDataWithIntendedValidatorHash(address validator, bytes memory data) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked("\x19\x00", validator, data));
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts-v4/interfaces/IERC1271.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (interfaces/IERC1271.sol)

pragma solidity ^0.8.0;

/**
 * @dev Interface of the ERC1271 standard signature validation method for
 * contracts as defined in https://eips.ethereum.org/EIPS/eip-1271[ERC-1271].
 *
 * _Available since v4.1._
 */
interface IERC1271 {
    /**
     * @dev Should return whether the signature provided is valid for the provided data
     * @param hash      Hash of the data to be signed
     * @param signature Signature byte array associated with _data
     */
    function isValidSignature(bytes32 hash, bytes memory signature) external view returns (bytes4 magicValue);
}


// ===== FILE: node_modules/_openzeppelin/contracts/utils/Bytes.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.5.0) (utils/Bytes.sol)

pragma solidity ^0.8.24;

import {Math} from "./math/Math.sol";

/**
 * @dev Bytes operations.
 */
library Bytes {
    /**
     * @dev Forward search for `s` in `buffer`
     * * If `s` is present in the buffer, returns the index of the first instance
     * * If `s` is not present in the buffer, returns type(uint256).max
     *
     * NOTE: replicates the behavior of https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/indexOf[Javascript's `Array.indexOf`]
     */
    function indexOf(bytes memory buffer, bytes1 s) internal pure returns (uint256) {
        return indexOf(buffer, s, 0);
    }

    /**
     * @dev Forward search for `s` in `buffer` starting at position `pos`
     * * If `s` is present in the buffer (at or after `pos`), returns the index of the next instance
     * * If `s` is not present in the buffer (at or after `pos`), returns type(uint256).max
     *
     * NOTE: replicates the behavior of https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/indexOf[Javascript's `Array.indexOf`]
     */
    function indexOf(bytes memory buffer, bytes1 s, uint256 pos) internal pure returns (uint256) {
        uint256 length = buffer.length;
        for (uint256 i = pos; i < length; ++i) {
            if (bytes1(_unsafeReadBytesOffset(buffer, i)) == s) {
                return i;
            }
        }
        return type(uint256).max;
    }

    /**
     * @dev Backward search for `s` in `buffer`
     * * If `s` is present in the buffer, returns the index of the last instance
     * * If `s` is not present in the buffer, returns type(uint256).max
     *
     * NOTE: replicates the behavior of https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/lastIndexOf[Javascript's `Array.lastIndexOf`]
     */
    function lastIndexOf(bytes memory buffer, bytes1 s) internal pure returns (uint256) {
        return lastIndexOf(buffer, s, type(uint256).max);
    }

    /**
     * @dev Backward search for `s` in `buffer` starting at position `pos`
     * * If `s` is present in the buffer (at or before `pos`), returns the index of the previous instance
     * * If `s` is not present in the buffer (at or before `pos`), returns type(uint256).max
     *
     * NOTE: replicates the behavior of https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/lastIndexOf[Javascript's `Array.lastIndexOf`]
     */
    function lastIndexOf(bytes memory buffer, bytes1 s, uint256 pos) internal pure returns (uint256) {
        unchecked {
            uint256 length = buffer.length;
            for (uint256 i = Math.min(Math.saturatingAdd(pos, 1), length); i > 0; --i) {
                if (bytes1(_unsafeReadBytesOffset(buffer, i - 1)) == s) {
                    return i - 1;
                }
            }
            return type(uint256).max;
        }
    }

    /**
     * @dev Copies the content of `buffer`, from `start` (included) to the end of `buffer` into a new bytes object in
     * memory.
     *
     * NOTE: replicates the behavior of https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/slice[Javascript's `Array.slice`]
     */
    function slice(bytes memory buffer, uint256 start) internal pure returns (bytes memory) {
        return slice(buffer, start, buffer.length);
    }

    /**
     * @dev Copies the content of `buffer`, from `start` (included) to `end` (excluded) into a new bytes object in
     * memory. The `end` argument is truncated to the length of the `buffer`.
     *
     * NOTE: replicates the behavior of https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/slice[Javascript's `Array.slice`]
     */
    function slice(bytes memory buffer, uint256 start, uint256 end) internal pure returns (bytes memory) {
        // sanitize
        end = Math.min(end, buffer.length);
        start = Math.min(start, end);

        // allocate and copy
        bytes memory result = new bytes(end - start);
        assembly ("memory-safe") {
            mcopy(add(result, 0x20), add(add(buffer, 0x20), start), sub(end, start))
        }

        return result;
    }

    /**
     * @dev Moves the content of `buffer`, from `start` (included) to the end of `buffer` to the start of that buffer.
     *
     * NOTE: This function modifies the provided buffer in place. If you need to preserve the original buffer, use {slice} instead
     * NOTE: replicates the behavior of https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/splice[Javascript's `Array.splice`]
     */
    function splice(bytes memory buffer, uint256 start) internal pure returns (bytes memory) {
        return splice(buffer, start, buffer.length);
    }

    /**
     * @dev Moves the content of `buffer`, from `start` (included) to end (excluded) to the start of that buffer. The
     * `end` argument is truncated to the length of the `buffer`.
     *
     * NOTE: This function modifies the provided buffer in place. If you need to preserve the original buffer, use {slice} instead
     * NOTE: replicates the behavior of https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/splice[Javascript's `Array.splice`]
     */
    function splice(bytes memory buffer, uint256 start, uint256 end) internal pure returns (bytes memory) {
        // sanitize
        end = Math.min(end, buffer.length);
        start = Math.min(start, end);

        // allocate and copy
        assembly ("memory-safe") {
            mcopy(add(buffer, 0x20), add(add(buffer, 0x20), start), sub(end, start))
            mstore(buffer, sub(end, start))
        }

        return buffer;
    }

    /**
     * @dev Concatenate an array of bytes into a single bytes object.
     *
     * For fixed bytes types, we recommend using the solidity built-in `bytes.concat` or (equivalent)
     * `abi.encodePacked`.
     *
     * NOTE: this could be done in assembly with a single loop that expands starting at the FMP, but that would be
     * significantly less readable. It might be worth benchmarking the savings of the full-assembly approach.
     */
    function concat(bytes[] memory buffers) internal pure returns (bytes memory) {
        uint256 length = 0;
        for (uint256 i = 0; i < buffers.length; ++i) {
            length += buffers[i].length;
        }

        bytes memory result = new bytes(length);

        uint256 offset = 0x20;
        for (uint256 i = 0; i < buffers.length; ++i) {
            bytes memory input = buffers[i];
            assembly ("memory-safe") {
                mcopy(add(result, offset), add(input, 0x20), mload(input))
            }
            unchecked {
                offset += input.length;
            }
        }

        return result;
    }

    /**
     * @dev Returns true if the two byte buffers are equal.
     */
    function equal(bytes memory a, bytes memory b) internal pure returns (bool) {
        return a.length == b.length && keccak256(a) == keccak256(b);
    }

    /**
     * @dev Reverses the byte order of a bytes32 value, converting between little-endian and big-endian.
     * Inspired by https://graphics.stanford.edu/~seander/bithacks.html#ReverseParallel[Reverse Parallel]
     */
    function reverseBytes32(bytes32 value) internal pure returns (bytes32) {
        value = // swap bytes
            ((value >> 8) & 0x00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF) |
            ((value & 0x00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF) << 8);
        value = // swap 2-byte long pairs
            ((value >> 16) & 0x0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF) |
            ((value & 0x0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF) << 16);
        value = // swap 4-byte long pairs
            ((value >> 32) & 0x00000000FFFFFFFF00000000FFFFFFFF00000000FFFFFFFF00000000FFFFFFFF) |
            ((value & 0x00000000FFFFFFFF00000000FFFFFFFF00000000FFFFFFFF00000000FFFFFFFF) << 32);
        value = // swap 8-byte long pairs
            ((value >> 64) & 0x0000000000000000FFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF) |
            ((value & 0x0000000000000000FFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF) << 64);
        return (value >> 128) | (value << 128); // swap 16-byte long pairs
    }

    /// @dev Same as {reverseBytes32} but optimized for 128-bit values.
    function reverseBytes16(bytes16 value) internal pure returns (bytes16) {
        value = // swap bytes
            ((value & 0xFF00FF00FF00FF00FF00FF00FF00FF00) >> 8) |
            ((value & 0x00FF00FF00FF00FF00FF00FF00FF00FF) << 8);
        value = // swap 2-byte long pairs
            ((value & 0xFFFF0000FFFF0000FFFF0000FFFF0000) >> 16) |
            ((value & 0x0000FFFF0000FFFF0000FFFF0000FFFF) << 16);
        value = // swap 4-byte long pairs
            ((value & 0xFFFFFFFF00000000FFFFFFFF00000000) >> 32) |
            ((value & 0x00000000FFFFFFFF00000000FFFFFFFF) << 32);
        return (value >> 64) | (value << 64); // swap 8-byte long pairs
    }

    /// @dev Same as {reverseBytes32} but optimized for 64-bit values.
    function reverseBytes8(bytes8 value) internal pure returns (bytes8) {
        value = ((value & 0xFF00FF00FF00FF00) >> 8) | ((value & 0x00FF00FF00FF00FF) << 8); // swap bytes
        value = ((value & 0xFFFF0000FFFF0000) >> 16) | ((value & 0x0000FFFF0000FFFF) << 16); // swap 2-byte long pairs
        return (value >> 32) | (value << 32); // swap 4-byte long pairs
    }

    /// @dev Same as {reverseBytes32} but optimized for 32-bit values.
    function reverseBytes4(bytes4 value) internal pure returns (bytes4) {
        value = ((value & 0xFF00FF00) >> 8) | ((value & 0x00FF00FF) << 8); // swap bytes
        return (value >> 16) | (value << 16); // swap 2-byte long pairs
    }

    /// @dev Same as {reverseBytes32} but optimized for 16-bit values.
    function reverseBytes2(bytes2 value) internal pure returns (bytes2) {
        return (value >> 8) | (value << 8);
    }

    /**
     * @dev Counts the number of leading zero bits a bytes array. Returns `8 * buffer.length`
     * if the buffer is all zeros.
     */
    function clz(bytes memory buffer) internal pure returns (uint256) {
        for (uint256 i = 0; i < buffer.length; i += 0x20) {
            bytes32 chunk = _unsafeReadBytesOffset(buffer, i);
            if (chunk != bytes32(0)) {
                return Math.min(8 * i + Math.clz(uint256(chunk)), 8 * buffer.length);
            }
        }
        return 8 * buffer.length;
    }

    /**
     * @dev Reads a bytes32 from a bytes array without bounds checking.
     *
     * NOTE: making this function internal would mean it could be used with memory unsafe offset, and marking the
     * assembly block as such would prevent some optimizations.
     */
    function _unsafeReadBytesOffset(bytes memory buffer, uint256 offset) private pure returns (bytes32 value) {
        // This is not memory safe in the general case, but all calls to this private function are within bounds.
        assembly ("memory-safe") {
            value := mload(add(add(buffer, 0x20), offset))
        }
    }
}


// ===== FILE: contracts/periphery/mintburn/AuthorizedFundedFlow.sol =====
// SPDX-License-Identifier: BUSL-1.1
pragma solidity ^0.8.23;

/// @notice Library shared between handler contracts and modules to communicate from handler to module what context is a
/// specific function being called in
abstract contract AuthorizedFundedFlow {
    // keccak256(abi.encode(uint256(keccak256("erc7201:AuthorizedFundedFlow.bool")) - 1)) & ~bytes32(uint256(0xff));
    bytes32 public constant AUTHORIZED_FUNDED_FLOW_SLOT =
        0xc56a3250645180a53cd9e196b2ee0a634a4f54e2edf59ea457f2083917e4d100;

    error FundedFlowNotAuthorized();

    modifier authorizeFundedFlow() {
        bytes32 slot = AUTHORIZED_FUNDED_FLOW_SLOT;
        assembly {
            sstore(slot, 1)
        }
        _;
        assembly {
            sstore(slot, 0)
        }
    }

    modifier onlyAuthorizedFlow() {
        bytes32 slot = AUTHORIZED_FUNDED_FLOW_SLOT;
        bool authorized;
        assembly {
            authorized := sload(slot)
        }
        if (!authorized) {
            revert FundedFlowNotAuthorized();
        }
        _;
    }
}


// ===== FILE: contracts/periphery/mintburn/HyperCoreFlowRoles.sol =====
//SPDX-License-Identifier: Unlicense
pragma solidity ^0.8.0;

abstract contract HyperCoreFlowRoles {
    bytes32 public constant PERMISSIONED_BOT_ROLE = keccak256("PERMISSIONED_BOT_ROLE");
    bytes32 public constant FUNDS_SWEEPER_ROLE = keccak256("FUNDS_SWEEPER_ROLE");
    bytes32 public constant DIRECT_CALLER_ROLE = keccak256("DIRECT_CALLER_ROLE");
}


// ===== FILE: node_modules/_openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (access/AccessControl.sol)

pragma solidity ^0.8.20;

import {IAccessControl} from "@openzeppelin/contracts/access/IAccessControl.sol";
import {ContextUpgradeable} from "../utils/ContextUpgradeable.sol";
import {IERC165} from "@openzeppelin/contracts/utils/introspection/IERC165.sol";
import {ERC165Upgradeable} from "../utils/introspection/ERC165Upgradeable.sol";
import {Initializable} from "@openzeppelin/contracts/proxy/utils/Initializable.sol";

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
    /// @inheritdoc IERC165
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
     * @dev Attempts to revoke `role` from `account` and returns a boolean indicating if `role` was revoked.
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


// ===== FILE: node_modules/_openzeppelin/contracts-v4/security/ReentrancyGuard.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.0) (security/ReentrancyGuard.sol)

pragma solidity ^0.8.0;

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
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;

    uint256 private _status;

    constructor() {
        _status = _NOT_ENTERED;
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
        // On the first call to nonReentrant, _status will be _NOT_ENTERED
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");

        // Any calls to nonReentrant after this point will fail
        _status = _ENTERED;
    }

    function _nonReentrantAfter() private {
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _status = _NOT_ENTERED;
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        return _status == _ENTERED;
    }
}


// ===== FILE: contracts/chain-adapters/DonationBox.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { AccessControl } from "@openzeppelin/contracts/access/AccessControl.sol";
import { IERC20 } from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import { SafeERC20 } from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @notice Users can donate tokens to this contract that only authorized withdrawers can withdraw.
 * @dev This contract is designed to be used as a convenience for storing funds to pay for
 * future transactions, such as donating custom gas tokens to pay for future retryable ticket messages
 * to be sent via the Arbitrum_Adapter.
 * @dev Multiple addresses can be granted the WITHDRAWER_ROLE to withdraw funds.
 * @custom:security-contact bugs@across.to
 */
contract DonationBox is AccessControl {
    using SafeERC20 for IERC20;

    /// @notice Role for addresses authorized to withdraw funds
    bytes32 public constant WITHDRAWER_ROLE = keccak256("WITHDRAWER_ROLE");

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
    }

    /**
     * @notice Withdraw tokens from the contract.
     * @dev Only callable by addresses with WITHDRAWER_ROLE.
     * @param token Token to withdraw.
     * @param amount Amount of tokens to withdraw.
     */
    function withdraw(IERC20 token, uint256 amount) external onlyRole(WITHDRAWER_ROLE) {
        token.safeTransfer(msg.sender, amount);
    }
}


// ===== FILE: contracts/libraries/HyperCoreLib.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { IERC20 } from "@openzeppelin/contracts-v4/token/ERC20/IERC20.sol";
import { SafeERC20 } from "@openzeppelin/contracts-v4/token/ERC20/utils/SafeERC20.sol";
import { SafeCast } from "@openzeppelin/contracts-v4/utils/math/SafeCast.sol";
import { ICoreDepositWallet } from "../external/interfaces/ICoreDepositWallet.sol";

interface ICoreWriter {
    function sendRawAction(bytes calldata data) external;
}

library HyperCoreLib {
    using SafeERC20 for IERC20;

    // Time-in-Force order types
    enum Tif {
        None, // invalid
        ALO, // Add Liquidity Only
        GTC, // Good-Till-Cancel
        IOC // Immediate-or-Cancel
    }

    struct SpotBalance {
        uint64 total;
        uint64 hold; // Unused in this implementation
        uint64 entryNtl; // Unused in this implementation
    }

    struct TokenInfo {
        string name;
        uint64[] spots;
        uint64 deployerTradingFeeShare;
        address deployer;
        address evmContract;
        uint8 szDecimals;
        uint8 weiDecimals;
        int8 evmExtraWeiDecimals;
    }

    struct CoreUserExists {
        bool exists;
    }

    // Base asset bridge addresses
    address public constant BASE_ASSET_BRIDGE_ADDRESS = 0x2000000000000000000000000000000000000000;
    uint256 public constant BASE_ASSET_BRIDGE_ADDRESS_UINT256 = uint256(uint160(BASE_ASSET_BRIDGE_ADDRESS));

    // Precompile addresses
    address public constant SPOT_BALANCE_PRECOMPILE_ADDRESS = 0x0000000000000000000000000000000000000801;
    address public constant SPOT_PX_PRECOMPILE_ADDRESS = 0x0000000000000000000000000000000000000808;
    address public constant CORE_USER_EXISTS_PRECOMPILE_ADDRESS = 0x0000000000000000000000000000000000000810;
    address public constant TOKEN_INFO_PRECOMPILE_ADDRESS = 0x000000000000000000000000000000000000080C;
    address public constant CORE_WRITER_PRECOMPILE_ADDRESS = 0x3333333333333333333333333333333333333333;

    // USDC
    address public constant USDC_CORE_DEPOSIT_WALLET_ADDRESS = 0x6B9E773128f453f5c2C60935Ee2DE2CBc5390A24;
    uint64 public constant USDC_CORE_INDEX = 0;

    // CoreWriter action headers
    bytes4 public constant LIMIT_ORDER_HEADER = 0x01000001; // version=1, action=1
    bytes4 public constant SPOT_SEND_HEADER = 0x01000006; // version=1, action=6
    bytes4 public constant CANCEL_BY_CLOID_HEADER = 0x0100000B; // version=1, action=11
    bytes4 public constant SEND_ASSET_TO_DEX_HEADER = 0x0100000D; // version=1, action=13

    // HyperCore protocol constants
    uint32 public constant CORE_SPOT_DEX_ID = type(uint32).max;

    // Errors
    error LimitPxIsZero();
    error OrderSizeIsZero();
    error InvalidTif();
    error SpotBalancePrecompileCallFailed();
    error CoreUserExistsPrecompileCallFailed();
    error TokenInfoPrecompileCallFailed();
    error SpotPxPrecompileCallFailed();
    error InsufficientAmountForAccountActivation();
    error MaximumEVMSendAmountTooLarge();

    /**
     * @notice Transfer `amountEVM` from HyperEVM to `to` on HyperCore.
     * @dev Returns the amount credited on Core in Core units (post conversion).
     * @param erc20EVMAddress The address of the ERC20 token on HyperEVM
     * @param erc20CoreIndex The HyperCore index id of the token to transfer
     * @param to The address to receive tokens on HyperCore
     * @param amountEVM The amount to transfer on HyperEVM
     * @param decimalDiff The decimal difference of evmDecimals - coreDecimals
     * @param destinationDex The destination DEX on HyperCore
     * @param accountActivationFeeCore Present if we have to pay for account activation of `to` (meaning we're reducing our send amount)
     * @return amountEVMSent The amount sent on HyperEVM
     * @return amountCoreToReceive The amount credited on Core in Core units (post conversion)
     */
    function transferERC20EVMToCore(
        address erc20EVMAddress,
        uint64 erc20CoreIndex,
        address to,
        uint256 amountEVM,
        int8 decimalDiff,
        uint32 destinationDex,
        uint64 accountActivationFeeCore
    ) internal returns (uint256 amountEVMSent, uint64 amountCoreToReceive) {
        // if the transfer amount exceeds the bridge balance, this wil revert
        (uint256 _amountEVMToSend, uint64 _amountCoreToReceive) = maximumEVMSendAmountToAmounts(amountEVM, decimalDiff);
        if (_amountCoreToReceive <= accountActivationFeeCore) revert InsufficientAmountForAccountActivation();

        if (_amountEVMToSend != 0) {
            if (erc20CoreIndex == USDC_CORE_INDEX) {
                // USDC flow takes care of account creation fee for us, we don't need to reduce `_amountEVMToSend`
                IERC20(erc20EVMAddress).forceApprove(USDC_CORE_DEPOSIT_WALLET_ADDRESS, _amountEVMToSend);
                if (to == address(this)) {
                    ICoreDepositWallet(USDC_CORE_DEPOSIT_WALLET_ADDRESS).deposit(_amountEVMToSend, destinationDex);
                } else {
                    ICoreDepositWallet(USDC_CORE_DEPOSIT_WALLET_ADDRESS).depositFor(
                        to,
                        _amountEVMToSend,
                        destinationDex
                    );
                }
            } else {
                IERC20(erc20EVMAddress).safeTransfer(toAssetBridgeAddress(erc20CoreIndex), _amountEVMToSend);
                // Transfer the tokens from this contract on HyperCore to the `to` address on HyperCore
                if (to != address(this) || destinationDex != CORE_SPOT_DEX_ID) {
                    transferERC20CoreToCore(
                        erc20CoreIndex,
                        to,
                        _amountCoreToReceive - accountActivationFeeCore,
                        CORE_SPOT_DEX_ID,
                        destinationDex
                    );
                }
            }
        }

        return (_amountEVMToSend, _amountCoreToReceive - accountActivationFeeCore);
    }

    /**
     * @notice Bridges `amountEVM` of `erc20` from this address on HyperEVM to this address on HyperCore on the Spot DEX.
     * @dev Returns the amount sent on HyperEVM and the amount credited on Core in Core units (post conversion).
     * @dev The decimal difference is evmDecimals - coreDecimals
     * @param erc20EVMAddress The address of the ERC20 token on HyperEVM
     * @param erc20CoreIndex The HyperCore index id of the token to transfer
     * @param amountEVM The amount to transfer on HyperEVM
     * @param decimalDiff The decimal difference of evmDecimals - coreDecimals
     * @return amountEVMSent The amount sent on HyperEVM
     * @return amountCoreToReceive The amount credited on Core in Core units (post conversion)
     */
    function transferERC20EVMToSelfOnSpot(
        address erc20EVMAddress,
        uint64 erc20CoreIndex,
        uint256 amountEVM,
        int8 decimalDiff
    ) internal returns (uint256 amountEVMSent, uint64 amountCoreToReceive) {
        (amountEVMSent, amountCoreToReceive) = transferERC20EVMToSelfOnCore(
            erc20EVMAddress,
            erc20CoreIndex,
            amountEVM,
            decimalDiff,
            CORE_SPOT_DEX_ID
        );
    }

    /**
     * @notice Bridges `amountEVM` of `erc20` from this address on HyperEVM to this address on HyperCore.
     * @dev Returns the amount credited on Core in Core units (post conversion).
     * @dev The decimal difference is evmDecimals - coreDecimals
     * @param erc20EVMAddress The address of the ERC20 token on HyperEVM
     * @param erc20CoreIndex The HyperCore index id of the token to transfer
     * @param amountEVM The amount to transfer on HyperEVM
     * @param decimalDiff The decimal difference of evmDecimals - coreDecimals
     * @param destinationDex The destination DEX on HyperCore
     * @return amountEVMSent The amount sent on HyperEVM
     * @return amountCoreToReceive The amount credited on Core in Core units (post conversion)
     */
    function transferERC20EVMToSelfOnCore(
        address erc20EVMAddress,
        uint64 erc20CoreIndex,
        uint256 amountEVM,
        int8 decimalDiff,
        uint32 destinationDex
    ) internal returns (uint256 amountEVMSent, uint64 amountCoreToReceive) {
        return
            transferERC20EVMToCore(
                erc20EVMAddress,
                erc20CoreIndex,
                address(this),
                amountEVM,
                decimalDiff,
                destinationDex,
                // Contracts that are using this function HAVE to have their accounts created already
                0
            );
    }

    /**
     * @notice Transfers tokens from this contract on HyperCore to the `to` address on HyperCore on the Spot DEX.
     * @param erc20CoreIndex The HyperCore index id of the token
     * @param to The address to receive tokens on HyperCore
     * @param amountCore The amount to transfer on HyperCore
     */
    function transferERC20SpotToSpot(uint64 erc20CoreIndex, address to, uint64 amountCore) internal {
        transferERC20CoreToCore(erc20CoreIndex, to, amountCore, CORE_SPOT_DEX_ID, CORE_SPOT_DEX_ID);
    }

    /**
     * @notice Transfers tokens from this contract on HyperCore to the `to` address on HyperCore
     * @param erc20CoreIndex The HyperCore index id of the token
     * @param to The address to receive tokens on HyperCore
     * @param amountCore The amount to transfer on HyperCore
     * @param sourceDex The source DEX on HyperCore
     * @param destinationDex The destination DEX on HyperCore
     */
    function transferERC20CoreToCore(
        uint64 erc20CoreIndex,
        address to,
        uint64 amountCore,
        uint32 sourceDex,
        uint32 destinationDex
    ) internal {
        bytes memory action;
        bytes memory payload;

        if (destinationDex == CORE_SPOT_DEX_ID && sourceDex == CORE_SPOT_DEX_ID) {
            action = abi.encode(to, erc20CoreIndex, amountCore);
            payload = abi.encodePacked(SPOT_SEND_HEADER, action);
        } else {
            action = abi.encode(to, address(0), sourceDex, destinationDex, erc20CoreIndex, amountCore);
            payload = abi.encodePacked(SEND_ASSET_TO_DEX_HEADER, action);
        }

        ICoreWriter(CORE_WRITER_PRECOMPILE_ADDRESS).sendRawAction(payload);
    }

    /**
     * @notice Activate a user account on HyperCore from HyperEVM.
     * @param erc20EVMAddress The address of the ERC20 token on HyperEVM.
     * @param erc20CoreIndex The HyperCore index id of the token to transfer.
     * @param user The address to activate on HyperCore.
     * @param amountEVM The amount to transfer on HyperEVM.
     */
    function activateCoreAccountFromEVM(
        address erc20EVMAddress,
        uint64 erc20CoreIndex,
        address user,
        uint256 amountEVM
    ) internal {
        if (erc20CoreIndex == USDC_CORE_INDEX) {
            IERC20(erc20EVMAddress).forceApprove(USDC_CORE_DEPOSIT_WALLET_ADDRESS, amountEVM);
            ICoreDepositWallet(USDC_CORE_DEPOSIT_WALLET_ADDRESS).depositFor(user, amountEVM, CORE_SPOT_DEX_ID);
        } else {
            IERC20(erc20EVMAddress).safeTransfer(toAssetBridgeAddress(erc20CoreIndex), amountEVM);
            // Transfer 1 wei to user on HyperCore to activate account
            transferERC20CoreToCore(erc20CoreIndex, user, 1, CORE_SPOT_DEX_ID, CORE_SPOT_DEX_ID);
        }
    }

    /**
     * @notice Submit a limit order on HyperCore.
     * @dev Expects price & size already scaled by 1e8 per HyperCore spec.
     * @param asset The asset index of the order
     * @param isBuy Whether the order is a buy order
     * @param limitPriceX1e8 The limit price of the order scaled by 1e8
     * @param sizeX1e8 The size of the order scaled by 1e8
     * @param reduceOnly If true, only reduce existing position rather than opening a new opposing order
     * @param tif Time-in-Force: ALO, GTC, IOC (None invalid)
     * @param cloid The client order id of the order, 0 means no cloid
     */
    function submitLimitOrder(
        uint32 asset,
        bool isBuy,
        uint64 limitPriceX1e8,
        uint64 sizeX1e8,
        bool reduceOnly,
        Tif tif,
        uint128 cloid
    ) internal {
        // Basic sanity checks
        if (limitPriceX1e8 == 0) revert LimitPxIsZero();
        if (sizeX1e8 == 0) revert OrderSizeIsZero();
        if (tif == Tif.None || uint8(tif) > uint8(type(Tif).max)) revert InvalidTif();

        // Encode the action
        bytes memory encodedAction = abi.encode(asset, isBuy, limitPriceX1e8, sizeX1e8, reduceOnly, uint8(tif), cloid);

        // Prefix with the limit-order header
        bytes memory data = abi.encodePacked(LIMIT_ORDER_HEADER, encodedAction);

        // Enqueue limit order to HyperCore via CoreWriter precompile
        ICoreWriter(CORE_WRITER_PRECOMPILE_ADDRESS).sendRawAction(data);
    }

    /**
     * @notice Enqueue a cancel-order-by-CLOID for a given asset.
     * @param asset The asset index of the order
     * @param cloid The client order id of the order
     */
    function cancelOrderByCloid(uint32 asset, uint128 cloid) internal {
        // Encode the action
        bytes memory encodedAction = abi.encode(asset, cloid);

        // Prefix with the cancel-by-cloid header
        bytes memory data = abi.encodePacked(CANCEL_BY_CLOID_HEADER, encodedAction);

        // Enqueue cancel order by CLOID to HyperCore via CoreWriter precompile
        ICoreWriter(CORE_WRITER_PRECOMPILE_ADDRESS).sendRawAction(data);
    }

    /**
     * @notice Get the balance of the specified ERC20 for `account` on HyperCore.
     * @param account The address of the account to get the balance of
     * @param token The token to get the balance of
     * @return balance The balance of the specified ERC20 for `account` on HyperCore
     */
    function spotBalance(address account, uint64 token) internal view returns (uint64 balance) {
        (bool success, bytes memory result) = SPOT_BALANCE_PRECOMPILE_ADDRESS.staticcall(abi.encode(account, token));
        if (!success) revert SpotBalancePrecompileCallFailed();
        SpotBalance memory _spotBalance = abi.decode(result, (SpotBalance));
        return _spotBalance.total;
    }

    /**
     * @notice Checks if the user exists / has been activated on HyperCore.
     * @param user The address of the user to check if they exist on HyperCore
     * @return exists True if the user exists on HyperCore, false otherwise
     */
    function coreUserExists(address user) internal view returns (bool) {
        (bool success, bytes memory result) = CORE_USER_EXISTS_PRECOMPILE_ADDRESS.staticcall(abi.encode(user));
        if (!success) revert CoreUserExistsPrecompileCallFailed();
        CoreUserExists memory _coreUserExists = abi.decode(result, (CoreUserExists));
        return _coreUserExists.exists;
    }

    /**
     * @notice Get the spot price of the specified asset on HyperCore.
     * @param index The asset index to get the spot price of
     * @return spotPx The spot price of the specified asset on HyperCore scaled by 1e8
     */
    function spotPx(uint32 index) internal view returns (uint64) {
        (bool success, bytes memory result) = SPOT_PX_PRECOMPILE_ADDRESS.staticcall(abi.encode(index));
        if (!success) revert SpotPxPrecompileCallFailed();
        return abi.decode(result, (uint64));
    }

    /**
     * @notice Get the info of the specified token on HyperCore.
     * @param erc20CoreIndex The token to get the info of
     * @return tokenInfo The info of the specified token on HyperCore
     */
    function tokenInfo(uint32 erc20CoreIndex) internal view returns (TokenInfo memory) {
        (bool success, bytes memory result) = TOKEN_INFO_PRECOMPILE_ADDRESS.staticcall(abi.encode(erc20CoreIndex));
        if (!success) revert TokenInfoPrecompileCallFailed();
        TokenInfo memory _tokenInfo = abi.decode(result, (TokenInfo));
        return _tokenInfo;
    }

    /**
     * @notice Checks if an amount is safe to bridge from HyperEVM to HyperCore
     * @dev Verifies that the asset bridge has sufficient balance to cover the amount plus a buffer
     * @param erc20CoreIndex The HyperCore index id of the token
     * @param coreAmount The amount that the bridging should result in on HyperCore
     * @param coreBufferAmount The minimum buffer amount that should remain on HyperCore after bridging
     * @return True if the bridge has enough balance to safely bridge the amount, false otherwise
     */
    function isCoreAmountSafeToBridge(
        uint64 erc20CoreIndex,
        uint64 coreAmount,
        uint64 coreBufferAmount
    ) internal view returns (bool) {
        address bridgeAddress = toAssetBridgeAddress(erc20CoreIndex);
        uint64 currentBridgeBalance = spotBalance(bridgeAddress, erc20CoreIndex);

        // Return true if currentBridgeBalance >= coreAmount + coreBufferAmount
        return currentBridgeBalance >= coreAmount + coreBufferAmount;
    }

    /**
     * @notice Converts a core index id to an asset bridge address
     * @param erc20CoreIndex The core token index id to convert
     * @return assetBridgeAddress The asset bridge address
     */
    function toAssetBridgeAddress(uint64 erc20CoreIndex) internal pure returns (address) {
        return address(uint160(BASE_ASSET_BRIDGE_ADDRESS_UINT256 + erc20CoreIndex));
    }

    /**
     * @notice Converts an asset bridge address to a core index id
     * @param assetBridgeAddress The asset bridge address to convert
     * @return erc20CoreIndex The core token index id
     */
    function toTokenId(address assetBridgeAddress) internal pure returns (uint64) {
        return uint64(uint160(assetBridgeAddress) - BASE_ASSET_BRIDGE_ADDRESS_UINT256);
    }

    /**
     * @notice Returns an amount to send on HyperEVM to receive AT LEAST the minimumCoreReceiveAmount on HyperCore
     * @param minimumCoreReceiveAmount The minimum amount desired to receive on HyperCore
     * @param decimalDiff The decimal difference of evmDecimals - coreDecimals
     * @return amountEVMToSend The amount to send on HyperEVM to receive at least minimumCoreReceiveAmount on HyperCore
     * @return amountCoreToReceive The amount that will be received on core if the amountEVMToSend is sent from HyperEVM
     */
    function minimumCoreReceiveAmountToAmounts(
        uint64 minimumCoreReceiveAmount,
        int8 decimalDiff
    ) internal pure returns (uint256 amountEVMToSend, uint64 amountCoreToReceive) {
        if (decimalDiff == 0) {
            // Same decimals between HyperEVM and HyperCore
            amountEVMToSend = uint256(minimumCoreReceiveAmount);
            amountCoreToReceive = minimumCoreReceiveAmount;
        } else if (decimalDiff > 0) {
            // EVM token has more decimals than Core
            // Scale up to represent the same value in higher-precision EVM units
            amountEVMToSend = uint256(minimumCoreReceiveAmount) * (10 ** uint8(decimalDiff));
            amountCoreToReceive = minimumCoreReceiveAmount;
        } else {
            // Core token has more decimals than EVM
            // Scale down, rounding UP to avoid shortfall on Core
            uint256 scaleDivisor = 10 ** uint8(-decimalDiff);
            amountEVMToSend = (uint256(minimumCoreReceiveAmount) + scaleDivisor - 1) / scaleDivisor; // ceil division
            amountCoreToReceive = uint64(amountEVMToSend * scaleDivisor);
        }
    }

    /**
     * @notice Converts a maximum EVM amount to send into an EVM amount to send to avoid loss to dust,
     * @notice and the corresponding amount that will be received on Core.
     * @param maximumEVMSendAmount The maximum amount to send on HyperEVM
     * @param decimalDiff The decimal difference of evmDecimals - coreDecimals
     * @return amountEVMToSend The amount to send on HyperEVM
     * @return amountCoreToReceive The amount that will be received on HyperCore if the amountEVMToSend is sent
     */
    function maximumEVMSendAmountToAmounts(
        uint256 maximumEVMSendAmount,
        int8 decimalDiff
    ) internal pure returns (uint256 amountEVMToSend, uint64 amountCoreToReceive) {
        uint256 amountCoreToReceive256;

        /// @dev HyperLiquid decimal conversion: Scale EVM (u256,evmDecimals) -> Core (u64,coreDecimals)
        if (decimalDiff == 0) {
            amountEVMToSend = maximumEVMSendAmount;
            amountCoreToReceive256 = amountEVMToSend;
        } else if (decimalDiff > 0) {
            // EVM token has more decimals than Core
            uint256 scale = 10 ** uint8(decimalDiff);
            amountEVMToSend = maximumEVMSendAmount - (maximumEVMSendAmount % scale); // Safe: dustAmount = maximumEVMSendAmount % scale, so dust <= maximumEVMSendAmount
            amountCoreToReceive256 = amountEVMToSend / scale;
        } else {
            // Core token has more decimals than EVM
            uint256 scale = 10 ** uint8(-1 * decimalDiff);
            amountEVMToSend = maximumEVMSendAmount;
            amountCoreToReceive256 = amountEVMToSend * scale;
        }

        amountCoreToReceive = SafeCast.toUint64(amountCoreToReceive256);
    }

    function convertCoreDecimalsSimple(
        uint64 amountDecimalsFrom,
        uint8 decimalsFrom,
        uint8 decimalsTo
    ) internal pure returns (uint64) {
        if (decimalsFrom == decimalsTo) {
            return amountDecimalsFrom;
        } else if (decimalsFrom < decimalsTo) {
            return uint64(amountDecimalsFrom * 10 ** (decimalsTo - decimalsFrom));
        } else {
            // round down
            return uint64(amountDecimalsFrom / 10 ** (decimalsFrom - decimalsTo));
        }
    }

    /**
     * @notice Builds a CoreTokenInfo struct by fetching token info from HyperCore and computing derived values.
     * @param coreIndex The index of the token on HyperCore.
     * @param canBeUsedForAccountActivation Whether this token can be used to pay for account activation.
     * @param accountActivationFeeCore The account activation fee in Core units.
     * @param bridgeSafetyBufferCore Bridge buffer for checking safety of bridging evm -> core. In core units.
     * @return coreTokenInfo The constructed CoreTokenInfo struct.
     */
    function buildCoreTokenInfo(
        uint32 coreIndex,
        bool canBeUsedForAccountActivation,
        uint64 accountActivationFeeCore,
        uint64 bridgeSafetyBufferCore
    ) internal view returns (CoreTokenInfo memory coreTokenInfo) {
        TokenInfo memory _tokenInfo = tokenInfo(coreIndex);
        (uint256 accountActivationFeeEVM, ) = minimumCoreReceiveAmountToAmounts(
            accountActivationFeeCore,
            _tokenInfo.evmExtraWeiDecimals
        );

        coreTokenInfo = CoreTokenInfo({
            tokenInfo: _tokenInfo,
            coreIndex: uint64(coreIndex),
            canBeUsedForAccountActivation: canBeUsedForAccountActivation,
            accountActivationFeeEVM: accountActivationFeeEVM,
            accountActivationFeeCore: accountActivationFeeCore,
            bridgeSafetyBufferCore: bridgeSafetyBufferCore
        });
    }
}

// Info about the token on HyperCore.
struct CoreTokenInfo {
    // The token info on HyperCore.
    HyperCoreLib.TokenInfo tokenInfo;
    // The HyperCore index id of the token.
    uint64 coreIndex;
    // Whether the token can be used for account activation fee.
    bool canBeUsedForAccountActivation;
    // The account activation fee for the token.
    uint256 accountActivationFeeEVM;
    // The account activation fee for the token on Core.
    uint64 accountActivationFeeCore;
    // Bridge buffer to use when checking safety of bridging evm -> core. In core units
    uint64 bridgeSafetyBufferCore;
}


// ===== FILE: contracts/periphery/mintburn/SwapHandler.sol =====
//SPDX-License-Identifier: Unlicense
pragma solidity ^0.8.0;
import { IERC20 } from "@openzeppelin/contracts-v4/token/ERC20/IERC20.sol";
import { SafeERC20 } from "@openzeppelin/contracts-v4/token/ERC20/utils/SafeERC20.sol";
import { HyperCoreLib } from "../../libraries/HyperCoreLib.sol";
import { FinalTokenInfo } from "./Structs.sol";

contract SwapHandler {
    // See https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint#asset
    uint32 private constant SPOT_MARKET_INDEX_OFFSET = 10_000;

    address public immutable parentHandler;
    using SafeERC20 for IERC20;

    constructor() {
        parentHandler = msg.sender;
    }

    modifier onlyParentHandler() {
        require(msg.sender == parentHandler, "Not parent handler");
        _;
    }

    function transferFundsToSelfOnCore(
        address erc20EVMAddress,
        uint64 erc20CoreIndex,
        uint256 amountEVM,
        int8 decimalDiff
    ) external onlyParentHandler {
        HyperCoreLib.transferERC20EVMToSelfOnSpot(erc20EVMAddress, erc20CoreIndex, amountEVM, decimalDiff);
    }

    function transferFundsToUserOnCore(
        uint64 erc20CoreIndex,
        address to,
        uint64 amountCore,
        uint32 destinationDex
    ) external onlyParentHandler {
        HyperCoreLib.transferERC20CoreToCore(
            erc20CoreIndex,
            to,
            amountCore,
            HyperCoreLib.CORE_SPOT_DEX_ID,
            destinationDex
        );
    }

    function submitSpotLimitOrder(
        FinalTokenInfo memory finalTokenInfo,
        uint64 limitPriceX1e8,
        uint64 sizeX1e8,
        uint128 cloid
    ) external onlyParentHandler {
        HyperCoreLib.submitLimitOrder(
            finalTokenInfo.spotIndex + SPOT_MARKET_INDEX_OFFSET,
            finalTokenInfo.isBuy,
            limitPriceX1e8,
            sizeX1e8,
            false,
            HyperCoreLib.Tif.GTC,
            cloid
        );
    }

    function cancelOrderByCloid(uint32 spotIndex, uint128 cloid) external onlyParentHandler {
        HyperCoreLib.cancelOrderByCloid(spotIndex + SPOT_MARKET_INDEX_OFFSET, cloid);
    }

    function sweepErc20(address token, uint256 amount) external onlyParentHandler {
        IERC20(token).safeTransfer(msg.sender, amount);
    }
}


// ===== FILE: contracts/periphery/mintburn/Constants.sol =====
// SPDX-License-Identifier: BUSL-1.1
pragma solidity ^0.8.0;

// Basis points decimals (4 decimals = 10000 for 100%)
uint256 constant BPS_DECIMALS = 4;

// Basis points scalar (10000 = 100%)
uint256 constant BPS_SCALAR = 10 ** BPS_DECIMALS;


// ===== FILE: contracts/handlers/MulticallHandler.sol =====
// SPDX-License-Identifier: GPL-3.0-only
pragma solidity ^0.8.0;

import "../interfaces/SpokePoolMessageHandler.sol";
import "@openzeppelin/contracts-v4/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts-v4/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts-v4/utils/Address.sol";
import "@openzeppelin/contracts-v4/security/ReentrancyGuard.sol";

/**
 * @title Across Multicall contract that allows a user to specify a series of calls that should be made by the handler
 * via the message field in the deposit.
 * @dev This contract makes the calls blindly. The caller should ensure that the tokens received by the handler are
 * completely consumed; otherwise leftover balances will be sent to the fallbackRecipient (when one is provided)
 * or remain on this contract.
 *
 * @dev This contract is a stateless utility with no per-user accounting or admin rescue. Tokens delivered to it
 * are expected to be consumed in the same transaction; any balances left on the contract can be claimed by any
 * caller.
 */
contract MulticallHandler is AcrossMessageHandler, ReentrancyGuard {
    using SafeERC20 for IERC20;
    using Address for address payable;

    struct Call {
        address target;
        bytes callData;
        uint256 value;
    }

    struct Replacement {
        address token;
        uint256 offset;
    }

    struct Instructions {
        //  Calls that will be attempted.
        Call[] calls;
        // Where the tokens go if any part of the call fails.
        // Leftover tokens are sent here as well if the action succeeds.
        address fallbackRecipient;
    }

    // Emitted when one of the calls fails. Note: all calls are reverted in this case.
    event CallsFailed(Call[] calls, address indexed fallbackRecipient);

    // Emitted when there are leftover tokens that are sent to the fallbackRecipient.
    event DrainedTokens(address indexed recipient, address indexed token, uint256 indexed amount);

    // Errors
    error CallReverted(uint256 index, Call[] calls);
    error NotSelf();
    error InvalidCall(uint256 index, Call[] calls);
    error ReplacementCallFailed(bytes callData);
    error CalldataTooShort(uint256 callDataLength, uint256 offset);

    modifier onlySelf() {
        _requireSelf();
        _;
    }

    /**
     * @notice Main entrypoint for the handler called by the SpokePool contract.
     * @dev This will execute all calls encoded in the msg. The caller is responsible for making sure all tokens are
     * drained from this contract by the end of the series of calls. If not, they can be stolen.
     * A drainLeftoverTokens call can be included as a way to drain any remaining tokens from this contract.
     * @param token The token address that was received from the relay
     * @param message abi encoded array of Call structs, containing a target, callData, and value for each call that
     * the contract should make.
     */
    function handleV3AcrossMessage(address token, uint256, address, bytes memory message) public virtual nonReentrant {
        Instructions memory instructions = abi.decode(message, (Instructions));

        // If there is no fallback recipient, call and revert if the inner call fails.
        if (instructions.fallbackRecipient == address(0)) {
            this.attemptCalls(instructions.calls);
            return;
        }

        // Otherwise, try the call and send to the fallback recipient if any tokens are leftover.
        (bool success, ) = address(this).call(abi.encodeCall(this.attemptCalls, (instructions.calls)));
        if (!success) emit CallsFailed(instructions.calls, instructions.fallbackRecipient);

        // If there are leftover tokens, send them to the fallback recipient regardless of execution success.
        _drainRemainingTokens(token, payable(instructions.fallbackRecipient));
    }

    function attemptCalls(Call[] memory calls) external onlySelf {
        uint256 length = calls.length;
        for (uint256 i = 0; i < length; ++i) {
            Call memory call = calls[i];

            // If we are calling an EOA with calldata, assume target was incorrectly specified and revert.
            if (call.callData.length > 0 && call.target.code.length == 0) {
                revert InvalidCall(i, calls);
            }

            (bool success, ) = call.target.call{ value: call.value }(call.callData);
            if (!success) revert CallReverted(i, calls);
        }
    }

    function drainLeftoverTokens(address token, address payable destination) external onlySelf {
        _drainRemainingTokens(token, destination);
    }

    function _drainRemainingTokens(address token, address payable destination) internal {
        if (token != address(0)) {
            // ERC20 token.
            uint256 amount = IERC20(token).balanceOf(address(this));
            if (amount > 0) {
                IERC20(token).safeTransfer(destination, amount);
                emit DrainedTokens(destination, token, amount);
            }
        } else {
            // Send native token
            uint256 amount = address(this).balance;
            if (amount > 0) {
                destination.sendValue(amount);
            }
        }
    }

    /**
     * @notice Executes a call while replacing specified calldata offsets with current token/native balances.
     * @dev Modifies calldata in-place using OR operations. Target calldata positions must be zeroed out.
     * Cannot handle negative balances, making it incompatible with DEXs requiring negative input amounts.
     * For native balance (token = address(0)), the entire balance is used as call value.
     * @param target The contract address to call
     * @param callData The calldata to execute, with zero values at replacement positions
     * @param value The native token value to send (ignored if native balance replacement is used)
     * @param replacement Array of Replacement structs specifying token addresses and byte offsets for balance injection
     */
    function makeCallWithBalance(
        address target,
        bytes memory callData,
        uint256 value,
        Replacement[] calldata replacement
    ) external onlySelf {
        for (uint256 i = 0; i < replacement.length; ++i) {
            uint256 bal = 0;
            if (replacement[i].token != address(0)) {
                bal = IERC20(replacement[i].token).balanceOf(address(this));
            } else {
                bal = address(this).balance;

                // If we're using the native balance, we assume that the caller wants to send the full value to the target.
                value = bal;
            }

            // + 32 to skip the length of the calldata
            uint256 offset = replacement[i].offset + 32;

            // 32 has already been added to the offset, and the replacement value is 32 bytes long, so
            // we don't need to add 32 here. We just directly compare the offset with the length of the calldata.
            if (offset > callData.length) revert CalldataTooShort(callData.length, offset);

            assembly ("memory-safe") {
                // Get the pointer to the offset that the caller wants to overwrite.
                let ptr := add(callData, offset)
                // Get the current value at the offset.
                let current := mload(ptr)
                // Or the current value with the new value.
                // Reasoning:
                // - caller should 0-out any portion that they want overwritten.
                // - if the caller is representing the balance in a smaller integer, like a uint160 or uint128,
                //   the higher bits will be 0 and not overwrite any other data in the calldata assuming
                //   the balance is small enough to fit in the smaller integer.
                // - The catch: the smaller integer where they want to store the balance must end no
                //   earlier than the 32nd byte in their calldata. Otherwise, this would require a
                //   negative offset, which is not possible.
                let val := or(bal, current)
                // Store the new value at the offset.
                mstore(ptr, val)
            }
        }

        (bool success, ) = target.call{ value: value }(callData);
        if (!success) revert ReplacementCallFailed(callData);
    }

    function _requireSelf() internal view {
        // Must be called by this contract to ensure that this cannot be triggered without the explicit consent of the
        // depositor (for a valid relay).
        if (msg.sender != address(this)) revert NotSelf();
    }

    // Used if the caller is trying to unwrap the native token to this contract.
    receive() external payable {}
}


// ===== FILE: node_modules/_openzeppelin/contracts/token/ERC20/IERC20.sol =====
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


// ===== FILE: node_modules/_openzeppelin/contracts/interfaces/IERC1363.sol =====
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


// ===== FILE: node_modules/_openzeppelin/contracts-v4/utils/Strings.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.0) (utils/Strings.sol)

pragma solidity ^0.8.0;

import "./math/Math.sol";
import "./math/SignedMath.sol";

/**
 * @dev String operations.
 */
library Strings {
    bytes16 private constant _SYMBOLS = "0123456789abcdef";
    uint8 private constant _ADDRESS_LENGTH = 20;

    /**
     * @dev Converts a `uint256` to its ASCII `string` decimal representation.
     */
    function toString(uint256 value) internal pure returns (string memory) {
        unchecked {
            uint256 length = Math.log10(value) + 1;
            string memory buffer = new string(length);
            uint256 ptr;
            /// @solidity memory-safe-assembly
            assembly {
                ptr := add(buffer, add(32, length))
            }
            while (true) {
                ptr--;
                /// @solidity memory-safe-assembly
                assembly {
                    mstore8(ptr, byte(mod(value, 10), _SYMBOLS))
                }
                value /= 10;
                if (value == 0) break;
            }
            return buffer;
        }
    }

    /**
     * @dev Converts a `int256` to its ASCII `string` decimal representation.
     */
    function toString(int256 value) internal pure returns (string memory) {
        return string(abi.encodePacked(value < 0 ? "-" : "", toString(SignedMath.abs(value))));
    }

    /**
     * @dev Converts a `uint256` to its ASCII `string` hexadecimal representation.
     */
    function toHexString(uint256 value) internal pure returns (string memory) {
        unchecked {
            return toHexString(value, Math.log256(value) + 1);
        }
    }

    /**
     * @dev Converts a `uint256` to its ASCII `string` hexadecimal representation with fixed length.
     */
    function toHexString(uint256 value, uint256 length) internal pure returns (string memory) {
        bytes memory buffer = new bytes(2 * length + 2);
        buffer[0] = "0";
        buffer[1] = "x";
        for (uint256 i = 2 * length + 1; i > 1; --i) {
            buffer[i] = _SYMBOLS[value & 0xf];
            value >>= 4;
        }
        require(value == 0, "Strings: hex length insufficient");
        return string(buffer);
    }

    /**
     * @dev Converts an `address` with fixed length of 20 bytes to its not checksummed ASCII `string` hexadecimal representation.
     */
    function toHexString(address addr) internal pure returns (string memory) {
        return toHexString(uint256(uint160(addr)), _ADDRESS_LENGTH);
    }

    /**
     * @dev Returns true if the two strings are equal.
     */
    function equal(string memory a, string memory b) internal pure returns (bool) {
        return keccak256(bytes(a)) == keccak256(bytes(b));
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts/utils/math/Math.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.5.0) (utils/math/Math.sol)

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
     * @dev Return the 512-bit addition of two uint256.
     *
     * The result is stored in two 256 variables such that sum = high * 2²⁵⁶ + low.
     */
    function add512(uint256 a, uint256 b) internal pure returns (uint256 high, uint256 low) {
        assembly ("memory-safe") {
            low := add(a, b)
            high := lt(low, a)
        }
    }

    /**
     * @dev Return the 512-bit multiplication of two uint256.
     *
     * The result is stored in two 256 variables such that product = high * 2²⁵⁶ + low.
     */
    function mul512(uint256 a, uint256 b) internal pure returns (uint256 high, uint256 low) {
        // 512-bit multiply [high low] = x * y. Compute the product mod 2²⁵⁶ and mod 2²⁵⁶ - 1, then use
        // the Chinese Remainder Theorem to reconstruct the 512 bit result. The result is stored in two 256
        // variables such that product = high * 2²⁵⁶ + low.
        assembly ("memory-safe") {
            let mm := mulmod(a, b, not(0))
            low := mul(a, b)
            high := sub(sub(mm, low), lt(mm, low))
        }
    }

    /**
     * @dev Returns the addition of two unsigned integers, with a success flag (no overflow).
     */
    function tryAdd(uint256 a, uint256 b) internal pure returns (bool success, uint256 result) {
        unchecked {
            uint256 c = a + b;
            success = c >= a;
            result = c * SafeCast.toUint(success);
        }
    }

    /**
     * @dev Returns the subtraction of two unsigned integers, with a success flag (no overflow).
     */
    function trySub(uint256 a, uint256 b) internal pure returns (bool success, uint256 result) {
        unchecked {
            uint256 c = a - b;
            success = c <= a;
            result = c * SafeCast.toUint(success);
        }
    }

    /**
     * @dev Returns the multiplication of two unsigned integers, with a success flag (no overflow).
     */
    function tryMul(uint256 a, uint256 b) internal pure returns (bool success, uint256 result) {
        unchecked {
            uint256 c = a * b;
            assembly ("memory-safe") {
                // Only true when the multiplication doesn't overflow
                // (c / a == b) || (a == 0)
                success := or(eq(div(c, a), b), iszero(a))
            }
            // equivalent to: success ? c : 0
            result = c * SafeCast.toUint(success);
        }
    }

    /**
     * @dev Returns the division of two unsigned integers, with a success flag (no division by zero).
     */
    function tryDiv(uint256 a, uint256 b) internal pure returns (bool success, uint256 result) {
        unchecked {
            success = b > 0;
            assembly ("memory-safe") {
                // The `DIV` opcode returns zero when the denominator is 0.
                result := div(a, b)
            }
        }
    }

    /**
     * @dev Returns the remainder of dividing two unsigned integers, with a success flag (no division by zero).
     */
    function tryMod(uint256 a, uint256 b) internal pure returns (bool success, uint256 result) {
        unchecked {
            success = b > 0;
            assembly ("memory-safe") {
                // The `MOD` opcode returns zero when the denominator is 0.
                result := mod(a, b)
            }
        }
    }

    /**
     * @dev Unsigned saturating addition, bounds to `2²⁵⁶ - 1` instead of overflowing.
     */
    function saturatingAdd(uint256 a, uint256 b) internal pure returns (uint256) {
        (bool success, uint256 result) = tryAdd(a, b);
        return ternary(success, result, type(uint256).max);
    }

    /**
     * @dev Unsigned saturating subtraction, bounds to zero instead of overflowing.
     */
    function saturatingSub(uint256 a, uint256 b) internal pure returns (uint256) {
        (, uint256 result) = trySub(a, b);
        return result;
    }

    /**
     * @dev Unsigned saturating multiplication, bounds to `2²⁵⁶ - 1` instead of overflowing.
     */
    function saturatingMul(uint256 a, uint256 b) internal pure returns (uint256) {
        (bool success, uint256 result) = tryMul(a, b);
        return ternary(success, result, type(uint256).max);
    }

    /**
     * @dev Branchless ternary evaluation for `condition ? a : b`. Gas costs are constant.
     *
     * IMPORTANT: This function may reduce bytecode size and consume less gas when used standalone.
     * However, the compiler may optimize Solidity ternary operations (i.e. `condition ? a : b`) to only compute
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
            (uint256 high, uint256 low) = mul512(x, y);

            // Handle non-overflow cases, 256 by 256 division.
            if (high == 0) {
                // Solidity will revert if denominator == 0, unlike the div opcode on its own.
                // The surrounding unchecked block does not change this fact.
                // See https://docs.soliditylang.org/en/latest/control-structures.html#checked-or-unchecked-arithmetic.
                return low / denominator;
            }

            // Make sure the result is less than 2²⁵⁶. Also prevents denominator == 0.
            if (denominator <= high) {
                Panic.panic(ternary(denominator == 0, Panic.DIVISION_BY_ZERO, Panic.UNDER_OVERFLOW));
            }

            ///////////////////////////////////////////////
            // 512 by 256 division.
            ///////////////////////////////////////////////

            // Make division exact by subtracting the remainder from [high low].
            uint256 remainder;
            assembly ("memory-safe") {
                // Compute remainder using mulmod.
                remainder := mulmod(x, y, denominator)

                // Subtract 256 bit number from 512 bit number.
                high := sub(high, gt(remainder, low))
                low := sub(low, remainder)
            }

            // Factor powers of two out of denominator and compute largest power of two divisor of denominator.
            // Always >= 1. See https://cs.stackexchange.com/q/138556/92363.

            uint256 twos = denominator & (0 - denominator);
            assembly ("memory-safe") {
                // Divide denominator by twos.
                denominator := div(denominator, twos)

                // Divide [high low] by twos.
                low := div(low, twos)

                // Flip twos such that it is 2²⁵⁶ / twos. If twos is zero, then it becomes one.
                twos := add(div(sub(0, twos), twos), 1)
            }

            // Shift in bits from high into low.
            low |= high * twos;

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
            // less than 2²⁵⁶, this is the final result. We don't need to compute the high bits of the result and high
            // is no longer required.
            result = low * inverse;
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
     * @dev Calculates floor(x * y >> n) with full precision. Throws if result overflows a uint256.
     */
    function mulShr(uint256 x, uint256 y, uint8 n) internal pure returns (uint256 result) {
        unchecked {
            (uint256 high, uint256 low) = mul512(x, y);
            if (high >= 1 << n) {
                Panic.panic(Panic.UNDER_OVERFLOW);
            }
            return (high << (256 - n)) | (low >> n);
        }
    }

    /**
     * @dev Calculates x * y >> n with full precision, following the selected rounding direction.
     */
    function mulShr(uint256 x, uint256 y, uint8 n, Rounding rounding) internal pure returns (uint256) {
        return mulShr(x, y, n) + SafeCast.toUint(unsignedRoundsUp(rounding) && mulmod(x, y, 1 << n) > 0);
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
    function log2(uint256 x) internal pure returns (uint256 r) {
        // If value has upper 128 bits set, log2 result is at least 128
        r = SafeCast.toUint(x > 0xffffffffffffffffffffffffffffffff) << 7;
        // If upper 64 bits of 128-bit half set, add 64 to result
        r |= SafeCast.toUint((x >> r) > 0xffffffffffffffff) << 6;
        // If upper 32 bits of 64-bit half set, add 32 to result
        r |= SafeCast.toUint((x >> r) > 0xffffffff) << 5;
        // If upper 16 bits of 32-bit half set, add 16 to result
        r |= SafeCast.toUint((x >> r) > 0xffff) << 4;
        // If upper 8 bits of 16-bit half set, add 8 to result
        r |= SafeCast.toUint((x >> r) > 0xff) << 3;
        // If upper 4 bits of 8-bit half set, add 4 to result
        r |= SafeCast.toUint((x >> r) > 0xf) << 2;

        // Shifts value right by the current result and use it as an index into this lookup table:
        //
        // | x (4 bits) |  index  | table[index] = MSB position |
        // |------------|---------|-----------------------------|
        // |    0000    |    0    |        table[0] = 0         |
        // |    0001    |    1    |        table[1] = 0         |
        // |    0010    |    2    |        table[2] = 1         |
        // |    0011    |    3    |        table[3] = 1         |
        // |    0100    |    4    |        table[4] = 2         |
        // |    0101    |    5    |        table[5] = 2         |
        // |    0110    |    6    |        table[6] = 2         |
        // |    0111    |    7    |        table[7] = 2         |
        // |    1000    |    8    |        table[8] = 3         |
        // |    1001    |    9    |        table[9] = 3         |
        // |    1010    |   10    |        table[10] = 3        |
        // |    1011    |   11    |        table[11] = 3        |
        // |    1100    |   12    |        table[12] = 3        |
        // |    1101    |   13    |        table[13] = 3        |
        // |    1110    |   14    |        table[14] = 3        |
        // |    1111    |   15    |        table[15] = 3        |
        //
        // The lookup table is represented as a 32-byte value with the MSB positions for 0-15 in the last 16 bytes.
        assembly ("memory-safe") {
            r := or(r, byte(shr(r, x), 0x0000010102020202030303030303030300000000000000000000000000000000))
        }
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
    function log256(uint256 x) internal pure returns (uint256 r) {
        // If value has upper 128 bits set, log2 result is at least 128
        r = SafeCast.toUint(x > 0xffffffffffffffffffffffffffffffff) << 7;
        // If upper 64 bits of 128-bit half set, add 64 to result
        r |= SafeCast.toUint((x >> r) > 0xffffffffffffffff) << 6;
        // If upper 32 bits of 64-bit half set, add 32 to result
        r |= SafeCast.toUint((x >> r) > 0xffffffff) << 5;
        // If upper 16 bits of 32-bit half set, add 16 to result
        r |= SafeCast.toUint((x >> r) > 0xffff) << 4;
        // Add 1 if upper 8 bits of 16-bit half set, and divide accumulated result by 8
        return (r >> 3) | SafeCast.toUint((x >> r) > 0xff);
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

    /**
     * @dev Counts the number of leading zero bits in a uint256.
     */
    function clz(uint256 x) internal pure returns (uint256) {
        return ternary(x == 0, 256, 255 - log2(x));
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts/access/IAccessControl.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (access/IAccessControl.sol)

pragma solidity >=0.8.4;

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
     * {RoleAdminChanged} not being emitted to signal this.
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


// ===== FILE: node_modules/_openzeppelin/contracts-upgradeable/utils/ContextUpgradeable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.1) (utils/Context.sol)

pragma solidity ^0.8.20;
import {Initializable} from "@openzeppelin/contracts/proxy/utils/Initializable.sol";

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


// ===== FILE: node_modules/_openzeppelin/contracts/utils/introspection/IERC165.sol =====
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


// ===== FILE: node_modules/_openzeppelin/contracts-upgradeable/utils/introspection/ERC165Upgradeable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (utils/introspection/ERC165.sol)

pragma solidity ^0.8.20;

import {IERC165} from "@openzeppelin/contracts/utils/introspection/IERC165.sol";
import {Initializable} from "@openzeppelin/contracts/proxy/utils/Initializable.sol";

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
    /// @inheritdoc IERC165
    function supportsInterface(bytes4 interfaceId) public view virtual returns (bool) {
        return interfaceId == type(IERC165).interfaceId;
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts/proxy/utils/Initializable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.3.0) (proxy/utils/Initializable.sol)

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
        // - construction: the contract is initialized at version 1 (no reinitialization) and the
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
     * @dev Pointer to storage slot. Allows integrators to override it with a custom storage location.
     *
     * NOTE: Consider following the ERC-7201 formula to derive storage locations.
     */
    function _initializableStorageSlot() internal pure virtual returns (bytes32) {
        return INITIALIZABLE_STORAGE;
    }

    /**
     * @dev Returns a pointer to the storage namespace.
     */
    // solhint-disable-next-line var-name-mixedcase
    function _getInitializableStorage() private pure returns (InitializableStorage storage $) {
        bytes32 slot = _initializableStorageSlot();
        assembly {
            $.slot := slot
        }
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts/access/AccessControl.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (access/AccessControl.sol)

pragma solidity ^0.8.20;

import {IAccessControl} from "./IAccessControl.sol";
import {Context} from "../utils/Context.sol";
import {IERC165, ERC165} from "../utils/introspection/ERC165.sol";

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
abstract contract AccessControl is Context, IAccessControl, ERC165 {
    struct RoleData {
        mapping(address account => bool) hasRole;
        bytes32 adminRole;
    }

    mapping(bytes32 role => RoleData) private _roles;

    bytes32 public constant DEFAULT_ADMIN_ROLE = 0x00;

    /**
     * @dev Modifier that checks that an account has a specific role. Reverts
     * with an {AccessControlUnauthorizedAccount} error including the required role.
     */
    modifier onlyRole(bytes32 role) {
        _checkRole(role);
        _;
    }

    /// @inheritdoc IERC165
    function supportsInterface(bytes4 interfaceId) public view virtual override returns (bool) {
        return interfaceId == type(IAccessControl).interfaceId || super.supportsInterface(interfaceId);
    }

    /**
     * @dev Returns `true` if `account` has been granted `role`.
     */
    function hasRole(bytes32 role, address account) public view virtual returns (bool) {
        return _roles[role].hasRole[account];
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
        return _roles[role].adminRole;
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
        bytes32 previousAdminRole = getRoleAdmin(role);
        _roles[role].adminRole = adminRole;
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
        if (!hasRole(role, account)) {
            _roles[role].hasRole[account] = true;
            emit RoleGranted(role, account, _msgSender());
            return true;
        } else {
            return false;
        }
    }

    /**
     * @dev Attempts to revoke `role` from `account` and returns a boolean indicating if `role` was revoked.
     *
     * Internal function without access restriction.
     *
     * May emit a {RoleRevoked} event.
     */
    function _revokeRole(bytes32 role, address account) internal virtual returns (bool) {
        if (hasRole(role, account)) {
            _roles[role].hasRole[account] = false;
            emit RoleRevoked(role, account, _msgSender());
            return true;
        } else {
            return false;
        }
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts-v4/utils/math/SafeCast.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.8.0) (utils/math/SafeCast.sol)
// This file was procedurally generated from scripts/generate/templates/SafeCast.js.

pragma solidity ^0.8.0;

/**
 * @dev Wrappers over Solidity's uintXX/intXX casting operators with added overflow
 * checks.
 *
 * Downcasting from uint256/int256 in Solidity does not revert on overflow. This can
 * easily result in undesired exploitation or bugs, since developers usually
 * assume that overflows raise errors. `SafeCast` restores this intuition by
 * reverting the transaction when such an operation overflows.
 *
 * Using this library instead of the unchecked operations eliminates an entire
 * class of bugs, so it's recommended to use it always.
 *
 * Can be combined with {SafeMath} and {SignedSafeMath} to extend it to smaller types, by performing
 * all math on `uint256` and `int256` and then downcasting.
 */
library SafeCast {
    /**
     * @dev Returns the downcasted uint248 from uint256, reverting on
     * overflow (when the input is greater than largest uint248).
     *
     * Counterpart to Solidity's `uint248` operator.
     *
     * Requirements:
     *
     * - input must fit into 248 bits
     *
     * _Available since v4.7._
     */
    function toUint248(uint256 value) internal pure returns (uint248) {
        require(value <= type(uint248).max, "SafeCast: value doesn't fit in 248 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint240(uint256 value) internal pure returns (uint240) {
        require(value <= type(uint240).max, "SafeCast: value doesn't fit in 240 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint232(uint256 value) internal pure returns (uint232) {
        require(value <= type(uint232).max, "SafeCast: value doesn't fit in 232 bits");
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
     *
     * _Available since v4.2._
     */
    function toUint224(uint256 value) internal pure returns (uint224) {
        require(value <= type(uint224).max, "SafeCast: value doesn't fit in 224 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint216(uint256 value) internal pure returns (uint216) {
        require(value <= type(uint216).max, "SafeCast: value doesn't fit in 216 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint208(uint256 value) internal pure returns (uint208) {
        require(value <= type(uint208).max, "SafeCast: value doesn't fit in 208 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint200(uint256 value) internal pure returns (uint200) {
        require(value <= type(uint200).max, "SafeCast: value doesn't fit in 200 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint192(uint256 value) internal pure returns (uint192) {
        require(value <= type(uint192).max, "SafeCast: value doesn't fit in 192 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint184(uint256 value) internal pure returns (uint184) {
        require(value <= type(uint184).max, "SafeCast: value doesn't fit in 184 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint176(uint256 value) internal pure returns (uint176) {
        require(value <= type(uint176).max, "SafeCast: value doesn't fit in 176 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint168(uint256 value) internal pure returns (uint168) {
        require(value <= type(uint168).max, "SafeCast: value doesn't fit in 168 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint160(uint256 value) internal pure returns (uint160) {
        require(value <= type(uint160).max, "SafeCast: value doesn't fit in 160 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint152(uint256 value) internal pure returns (uint152) {
        require(value <= type(uint152).max, "SafeCast: value doesn't fit in 152 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint144(uint256 value) internal pure returns (uint144) {
        require(value <= type(uint144).max, "SafeCast: value doesn't fit in 144 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint136(uint256 value) internal pure returns (uint136) {
        require(value <= type(uint136).max, "SafeCast: value doesn't fit in 136 bits");
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
     *
     * _Available since v2.5._
     */
    function toUint128(uint256 value) internal pure returns (uint128) {
        require(value <= type(uint128).max, "SafeCast: value doesn't fit in 128 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint120(uint256 value) internal pure returns (uint120) {
        require(value <= type(uint120).max, "SafeCast: value doesn't fit in 120 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint112(uint256 value) internal pure returns (uint112) {
        require(value <= type(uint112).max, "SafeCast: value doesn't fit in 112 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint104(uint256 value) internal pure returns (uint104) {
        require(value <= type(uint104).max, "SafeCast: value doesn't fit in 104 bits");
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
     *
     * _Available since v4.2._
     */
    function toUint96(uint256 value) internal pure returns (uint96) {
        require(value <= type(uint96).max, "SafeCast: value doesn't fit in 96 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint88(uint256 value) internal pure returns (uint88) {
        require(value <= type(uint88).max, "SafeCast: value doesn't fit in 88 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint80(uint256 value) internal pure returns (uint80) {
        require(value <= type(uint80).max, "SafeCast: value doesn't fit in 80 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint72(uint256 value) internal pure returns (uint72) {
        require(value <= type(uint72).max, "SafeCast: value doesn't fit in 72 bits");
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
     *
     * _Available since v2.5._
     */
    function toUint64(uint256 value) internal pure returns (uint64) {
        require(value <= type(uint64).max, "SafeCast: value doesn't fit in 64 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint56(uint256 value) internal pure returns (uint56) {
        require(value <= type(uint56).max, "SafeCast: value doesn't fit in 56 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint48(uint256 value) internal pure returns (uint48) {
        require(value <= type(uint48).max, "SafeCast: value doesn't fit in 48 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint40(uint256 value) internal pure returns (uint40) {
        require(value <= type(uint40).max, "SafeCast: value doesn't fit in 40 bits");
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
     *
     * _Available since v2.5._
     */
    function toUint32(uint256 value) internal pure returns (uint32) {
        require(value <= type(uint32).max, "SafeCast: value doesn't fit in 32 bits");
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
     *
     * _Available since v4.7._
     */
    function toUint24(uint256 value) internal pure returns (uint24) {
        require(value <= type(uint24).max, "SafeCast: value doesn't fit in 24 bits");
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
     *
     * _Available since v2.5._
     */
    function toUint16(uint256 value) internal pure returns (uint16) {
        require(value <= type(uint16).max, "SafeCast: value doesn't fit in 16 bits");
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
     *
     * _Available since v2.5._
     */
    function toUint8(uint256 value) internal pure returns (uint8) {
        require(value <= type(uint8).max, "SafeCast: value doesn't fit in 8 bits");
        return uint8(value);
    }

    /**
     * @dev Converts a signed int256 into an unsigned uint256.
     *
     * Requirements:
     *
     * - input must be greater than or equal to 0.
     *
     * _Available since v3.0._
     */
    function toUint256(int256 value) internal pure returns (uint256) {
        require(value >= 0, "SafeCast: value must be positive");
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
     *
     * _Available since v4.7._
     */
    function toInt248(int256 value) internal pure returns (int248 downcasted) {
        downcasted = int248(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 248 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt240(int256 value) internal pure returns (int240 downcasted) {
        downcasted = int240(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 240 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt232(int256 value) internal pure returns (int232 downcasted) {
        downcasted = int232(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 232 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt224(int256 value) internal pure returns (int224 downcasted) {
        downcasted = int224(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 224 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt216(int256 value) internal pure returns (int216 downcasted) {
        downcasted = int216(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 216 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt208(int256 value) internal pure returns (int208 downcasted) {
        downcasted = int208(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 208 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt200(int256 value) internal pure returns (int200 downcasted) {
        downcasted = int200(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 200 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt192(int256 value) internal pure returns (int192 downcasted) {
        downcasted = int192(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 192 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt184(int256 value) internal pure returns (int184 downcasted) {
        downcasted = int184(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 184 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt176(int256 value) internal pure returns (int176 downcasted) {
        downcasted = int176(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 176 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt168(int256 value) internal pure returns (int168 downcasted) {
        downcasted = int168(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 168 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt160(int256 value) internal pure returns (int160 downcasted) {
        downcasted = int160(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 160 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt152(int256 value) internal pure returns (int152 downcasted) {
        downcasted = int152(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 152 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt144(int256 value) internal pure returns (int144 downcasted) {
        downcasted = int144(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 144 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt136(int256 value) internal pure returns (int136 downcasted) {
        downcasted = int136(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 136 bits");
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
     *
     * _Available since v3.1._
     */
    function toInt128(int256 value) internal pure returns (int128 downcasted) {
        downcasted = int128(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 128 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt120(int256 value) internal pure returns (int120 downcasted) {
        downcasted = int120(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 120 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt112(int256 value) internal pure returns (int112 downcasted) {
        downcasted = int112(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 112 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt104(int256 value) internal pure returns (int104 downcasted) {
        downcasted = int104(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 104 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt96(int256 value) internal pure returns (int96 downcasted) {
        downcasted = int96(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 96 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt88(int256 value) internal pure returns (int88 downcasted) {
        downcasted = int88(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 88 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt80(int256 value) internal pure returns (int80 downcasted) {
        downcasted = int80(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 80 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt72(int256 value) internal pure returns (int72 downcasted) {
        downcasted = int72(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 72 bits");
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
     *
     * _Available since v3.1._
     */
    function toInt64(int256 value) internal pure returns (int64 downcasted) {
        downcasted = int64(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 64 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt56(int256 value) internal pure returns (int56 downcasted) {
        downcasted = int56(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 56 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt48(int256 value) internal pure returns (int48 downcasted) {
        downcasted = int48(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 48 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt40(int256 value) internal pure returns (int40 downcasted) {
        downcasted = int40(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 40 bits");
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
     *
     * _Available since v3.1._
     */
    function toInt32(int256 value) internal pure returns (int32 downcasted) {
        downcasted = int32(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 32 bits");
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
     *
     * _Available since v4.7._
     */
    function toInt24(int256 value) internal pure returns (int24 downcasted) {
        downcasted = int24(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 24 bits");
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
     *
     * _Available since v3.1._
     */
    function toInt16(int256 value) internal pure returns (int16 downcasted) {
        downcasted = int16(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 16 bits");
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
     *
     * _Available since v3.1._
     */
    function toInt8(int256 value) internal pure returns (int8 downcasted) {
        downcasted = int8(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 8 bits");
    }

    /**
     * @dev Converts an unsigned uint256 into a signed int256.
     *
     * Requirements:
     *
     * - input must be less than or equal to maxInt256.
     *
     * _Available since v3.0._
     */
    function toInt256(uint256 value) internal pure returns (int256) {
        // Note: Unsafe cast below is okay because `type(int256).max` is guaranteed to be positive
        require(value <= uint256(type(int256).max), "SafeCast: value doesn't fit in an int256");
        return int256(value);
    }
}


// ===== FILE: contracts/external/interfaces/ICoreDepositWallet.sol =====
/*
 * Copyright 2025 Circle Internet Group, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
pragma solidity ^0.8.0;

/**
 * @title IForwardDepositReceiver
 * @notice Interface for a contract that can receive deposits from the CCTP Forwarder
 * @dev Source: https://github.com/circlefin/hyperevm-circle-contracts/blob/master/src/interfaces/IForwardDepositReceiver.sol
 */
interface IForwardDepositReceiver {
    /**
     * @notice Deposit tokens for a recipient
     * @param recipient Recipient of the deposit
     * @param amount Amount of tokens to deposit
     * @param destinationId Forwarding-address-specific id used in conjunction with
     * recipient to route the deposit to a specific location.
     */
    function depositFor(address recipient, uint256 amount, uint32 destinationId) external;
}

/**
 * @title ICoreDepositWallet
 * @notice Minimal useful interface for the core deposit wallet
 * @dev Source: https://github.com/circlefin/hyperevm-circle-contracts/blob/master/src/interfaces/ICoreDepositWallet.sol
 */
interface ICoreDepositWallet is IForwardDepositReceiver {
    /**
     * @notice Deposits tokens for the sender.
     * @param amount The amount of tokens being deposited.
     * @param destinationDex The destination dex on HyperCore.
     */
    function deposit(uint256 amount, uint32 destinationDex) external;
}


// ===== FILE: contracts/interfaces/SpokePoolMessageHandler.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// This interface is expected to be implemented by any contract that expects to receive messages from the SpokePool.
interface AcrossMessageHandler {
    function handleV3AcrossMessage(address tokenSent, uint256 amount, address relayer, bytes memory message) external;
}


// ===== FILE: node_modules/_openzeppelin/contracts/interfaces/IERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (interfaces/IERC20.sol)

pragma solidity >=0.4.16;

import {IERC20} from "../token/ERC20/IERC20.sol";


// ===== FILE: node_modules/_openzeppelin/contracts/interfaces/IERC165.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (interfaces/IERC165.sol)

pragma solidity >=0.4.16;

import {IERC165} from "../utils/introspection/IERC165.sol";


// ===== FILE: node_modules/_openzeppelin/contracts-v4/utils/math/Math.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.0) (utils/math/Math.sol)

pragma solidity ^0.8.0;

/**
 * @dev Standard math utilities missing in the Solidity language.
 */
library Math {
    enum Rounding {
        Down, // Toward negative infinity
        Up, // Toward infinity
        Zero // Toward zero
    }

    /**
     * @dev Returns the largest of two numbers.
     */
    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a > b ? a : b;
    }

    /**
     * @dev Returns the smallest of two numbers.
     */
    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
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
     * This differs from standard division with `/` in that it rounds up instead
     * of rounding down.
     */
    function ceilDiv(uint256 a, uint256 b) internal pure returns (uint256) {
        // (a + b - 1) / b can overflow on addition, so we distribute.
        return a == 0 ? 0 : (a - 1) / b + 1;
    }

    /**
     * @notice Calculates floor(x * y / denominator) with full precision. Throws if result overflows a uint256 or denominator == 0
     * @dev Original credit to Remco Bloemen under MIT license (https://xn--2-umb.com/21/muldiv)
     * with further edits by Uniswap Labs also under MIT license.
     */
    function mulDiv(uint256 x, uint256 y, uint256 denominator) internal pure returns (uint256 result) {
        unchecked {
            // 512-bit multiply [prod1 prod0] = x * y. Compute the product mod 2^256 and mod 2^256 - 1, then use
            // use the Chinese Remainder Theorem to reconstruct the 512 bit result. The result is stored in two 256
            // variables such that product = prod1 * 2^256 + prod0.
            uint256 prod0; // Least significant 256 bits of the product
            uint256 prod1; // Most significant 256 bits of the product
            assembly {
                let mm := mulmod(x, y, not(0))
                prod0 := mul(x, y)
                prod1 := sub(sub(mm, prod0), lt(mm, prod0))
            }

            // Handle non-overflow cases, 256 by 256 division.
            if (prod1 == 0) {
                // Solidity will revert if denominator == 0, unlike the div opcode on its own.
                // The surrounding unchecked block does not change this fact.
                // See https://docs.soliditylang.org/en/latest/control-structures.html#checked-or-unchecked-arithmetic.
                return prod0 / denominator;
            }

            // Make sure the result is less than 2^256. Also prevents denominator == 0.
            require(denominator > prod1, "Math: mulDiv overflow");

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

            // Factor powers of two out of denominator and compute largest power of two divisor of denominator. Always >= 1.
            // See https://cs.stackexchange.com/q/138556/92363.

            // Does not overflow because the denominator cannot be zero at this stage in the function.
            uint256 twos = denominator & (~denominator + 1);
            assembly {
                // Divide denominator by twos.
                denominator := div(denominator, twos)

                // Divide [prod1 prod0] by twos.
                prod0 := div(prod0, twos)

                // Flip twos such that it is 2^256 / twos. If twos is zero, then it becomes one.
                twos := add(div(sub(0, twos), twos), 1)
            }

            // Shift in bits from prod1 into prod0.
            prod0 |= prod1 * twos;

            // Invert denominator mod 2^256. Now that denominator is an odd number, it has an inverse modulo 2^256 such
            // that denominator * inv = 1 mod 2^256. Compute the inverse by starting with a seed that is correct for
            // four bits. That is, denominator * inv = 1 mod 2^4.
            uint256 inverse = (3 * denominator) ^ 2;

            // Use the Newton-Raphson iteration to improve the precision. Thanks to Hensel's lifting lemma, this also works
            // in modular arithmetic, doubling the correct bits in each step.
            inverse *= 2 - denominator * inverse; // inverse mod 2^8
            inverse *= 2 - denominator * inverse; // inverse mod 2^16
            inverse *= 2 - denominator * inverse; // inverse mod 2^32
            inverse *= 2 - denominator * inverse; // inverse mod 2^64
            inverse *= 2 - denominator * inverse; // inverse mod 2^128
            inverse *= 2 - denominator * inverse; // inverse mod 2^256

            // Because the division is now exact we can divide by multiplying with the modular inverse of denominator.
            // This will give us the correct result modulo 2^256. Since the preconditions guarantee that the outcome is
            // less than 2^256, this is the final result. We don't need to compute the high bits of the result and prod1
            // is no longer required.
            result = prod0 * inverse;
            return result;
        }
    }

    /**
     * @notice Calculates x * y / denominator with full precision, following the selected rounding direction.
     */
    function mulDiv(uint256 x, uint256 y, uint256 denominator, Rounding rounding) internal pure returns (uint256) {
        uint256 result = mulDiv(x, y, denominator);
        if (rounding == Rounding.Up && mulmod(x, y, denominator) > 0) {
            result += 1;
        }
        return result;
    }

    /**
     * @dev Returns the square root of a number. If the number is not a perfect square, the value is rounded down.
     *
     * Inspired by Henry S. Warren, Jr.'s "Hacker's Delight" (Chapter 11).
     */
    function sqrt(uint256 a) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }

        // For our first guess, we get the biggest power of 2 which is smaller than the square root of the target.
        //
        // We know that the "msb" (most significant bit) of our target number `a` is a power of 2 such that we have
        // `msb(a) <= a < 2*msb(a)`. This value can be written `msb(a)=2**k` with `k=log2(a)`.
        //
        // This can be rewritten `2**log2(a) <= a < 2**(log2(a) + 1)`
        // → `sqrt(2**k) <= sqrt(a) < sqrt(2**(k+1))`
        // → `2**(k/2) <= sqrt(a) < 2**((k+1)/2) <= 2**(k/2 + 1)`
        //
        // Consequently, `2**(log2(a) / 2)` is a good first approximation of `sqrt(a)` with at least 1 correct bit.
        uint256 result = 1 << (log2(a) >> 1);

        // At this point `result` is an estimation with one bit of precision. We know the true value is a uint128,
        // since it is the square root of a uint256. Newton's method converges quadratically (precision doubles at
        // every iteration). We thus need at most 7 iteration to turn our partial result with one bit of precision
        // into the expected uint128 result.
        unchecked {
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            return min(result, a / result);
        }
    }

    /**
     * @notice Calculates sqrt(a), following the selected rounding direction.
     */
    function sqrt(uint256 a, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = sqrt(a);
            return result + (rounding == Rounding.Up && result * result < a ? 1 : 0);
        }
    }

    /**
     * @dev Return the log in base 2, rounded down, of a positive value.
     * Returns 0 if given 0.
     */
    function log2(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        unchecked {
            if (value >> 128 > 0) {
                value >>= 128;
                result += 128;
            }
            if (value >> 64 > 0) {
                value >>= 64;
                result += 64;
            }
            if (value >> 32 > 0) {
                value >>= 32;
                result += 32;
            }
            if (value >> 16 > 0) {
                value >>= 16;
                result += 16;
            }
            if (value >> 8 > 0) {
                value >>= 8;
                result += 8;
            }
            if (value >> 4 > 0) {
                value >>= 4;
                result += 4;
            }
            if (value >> 2 > 0) {
                value >>= 2;
                result += 2;
            }
            if (value >> 1 > 0) {
                result += 1;
            }
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
            return result + (rounding == Rounding.Up && 1 << result < value ? 1 : 0);
        }
    }

    /**
     * @dev Return the log in base 10, rounded down, of a positive value.
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
            return result + (rounding == Rounding.Up && 10 ** result < value ? 1 : 0);
        }
    }

    /**
     * @dev Return the log in base 256, rounded down, of a positive value.
     * Returns 0 if given 0.
     *
     * Adding one to the result gives the number of pairs of hex symbols needed to represent `value` as a hex string.
     */
    function log256(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        unchecked {
            if (value >> 128 > 0) {
                value >>= 128;
                result += 16;
            }
            if (value >> 64 > 0) {
                value >>= 64;
                result += 8;
            }
            if (value >> 32 > 0) {
                value >>= 32;
                result += 4;
            }
            if (value >> 16 > 0) {
                value >>= 16;
                result += 2;
            }
            if (value >> 8 > 0) {
                result += 1;
            }
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
            return result + (rounding == Rounding.Up && 1 << (result << 3) < value ? 1 : 0);
        }
    }
}


// ===== FILE: node_modules/_openzeppelin/contracts-v4/utils/math/SignedMath.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.8.0) (utils/math/SignedMath.sol)

pragma solidity ^0.8.0;

/**
 * @dev Standard signed math utilities missing in the Solidity language.
 */
library SignedMath {
    /**
     * @dev Returns the largest of two signed numbers.
     */
    function max(int256 a, int256 b) internal pure returns (int256) {
        return a > b ? a : b;
    }

    /**
     * @dev Returns the smallest of two signed numbers.
     */
    function min(int256 a, int256 b) internal pure returns (int256) {
        return a < b ? a : b;
    }

    /**
     * @dev Returns the average of two signed numbers without overflow.
     * The result is rounded towards zero.
     */
    function average(int256 a, int256 b) internal pure returns (int256) {
        // Formula from the book "Hacker's Delight"
        int256 x = (a & b) + ((a ^ b) >> 1);
        return x + (int256(uint256(x) >> 255) & (a ^ b));
    }

    /**
     * @dev Returns the absolute unsigned value of a signed value.
     */
    function abs(int256 n) internal pure returns (uint256) {
        unchecked {
            // must be unchecked in order to support `n = type(int256).min`
            return uint256(n >= 0 ? n : -n);
        }
    }
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


// ===== FILE: node_modules/_openzeppelin/contracts/utils/introspection/ERC165.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (utils/introspection/ERC165.sol)

pragma solidity ^0.8.20;

import {IERC165} from "./IERC165.sol";

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
abstract contract ERC165 is IERC165 {
    /// @inheritdoc IERC165
    function supportsInterface(bytes4 interfaceId) public view virtual returns (bool) {
        return interfaceId == type(IERC165).interfaceId;
    }
}
