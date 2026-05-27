// ===== FILE: _openzeppelin/contracts/interfaces/draft-IERC6093.sol =====
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


// ===== FILE: _openzeppelin/contracts/interfaces/IERC1363.sol =====
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


// ===== FILE: _openzeppelin/contracts/interfaces/IERC165.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (interfaces/IERC165.sol)

pragma solidity ^0.8.20;

import {IERC165} from "../utils/introspection/IERC165.sol";


// ===== FILE: _openzeppelin/contracts/interfaces/IERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (interfaces/IERC20.sol)

pragma solidity ^0.8.20;

import {IERC20} from "../token/ERC20/IERC20.sol";


// ===== FILE: _openzeppelin/contracts/token/ERC20/ERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.3.0) (token/ERC20/ERC20.sol)

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

    /**
     * @dev See {IERC20-totalSupply}.
     */
    function totalSupply() public view virtual returns (uint256) {
        return _totalSupply;
    }

    /**
     * @dev See {IERC20-balanceOf}.
     */
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

    /**
     * @dev See {IERC20-allowance}.
     */
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


// ===== FILE: _openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol =====
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


// ===== FILE: _openzeppelin/contracts/token/ERC20/IERC20.sol =====
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


// ===== FILE: _openzeppelin/contracts/utils/Context.sol =====
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


// ===== FILE: _openzeppelin/contracts/utils/Create2.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/Create2.sol)

pragma solidity ^0.8.20;

import {Errors} from "./Errors.sol";

/**
 * @dev Helper to make usage of the `CREATE2` EVM opcode easier and safer.
 * `CREATE2` can be used to compute in advance the address where a smart
 * contract will be deployed, which allows for interesting new mechanisms known
 * as 'counterfactual interactions'.
 *
 * See the https://eips.ethereum.org/EIPS/eip-1014#motivation[EIP] for more
 * information.
 */
library Create2 {
    /**
     * @dev There's no code to deploy.
     */
    error Create2EmptyBytecode();

    /**
     * @dev Deploys a contract using `CREATE2`. The address where the contract
     * will be deployed can be known in advance via {computeAddress}.
     *
     * The bytecode for a contract can be obtained from Solidity with
     * `type(contractName).creationCode`.
     *
     * Requirements:
     *
     * - `bytecode` must not be empty.
     * - `salt` must have not been used for `bytecode` already.
     * - the factory must have a balance of at least `amount`.
     * - if `amount` is non-zero, `bytecode` must have a `payable` constructor.
     */
    function deploy(uint256 amount, bytes32 salt, bytes memory bytecode) internal returns (address addr) {
        if (address(this).balance < amount) {
            revert Errors.InsufficientBalance(address(this).balance, amount);
        }
        if (bytecode.length == 0) {
            revert Create2EmptyBytecode();
        }
        assembly ("memory-safe") {
            addr := create2(amount, add(bytecode, 0x20), mload(bytecode), salt)
            // if no address was created, and returndata is not empty, bubble revert
            if and(iszero(addr), not(iszero(returndatasize()))) {
                let p := mload(0x40)
                returndatacopy(p, 0, returndatasize())
                revert(p, returndatasize())
            }
        }
        if (addr == address(0)) {
            revert Errors.FailedDeployment();
        }
    }

    /**
     * @dev Returns the address where a contract will be stored if deployed via {deploy}. Any change in the
     * `bytecodeHash` or `salt` will result in a new destination address.
     */
    function computeAddress(bytes32 salt, bytes32 bytecodeHash) internal view returns (address) {
        return computeAddress(salt, bytecodeHash, address(this));
    }

    /**
     * @dev Returns the address where a contract will be stored if deployed via {deploy} from a contract located at
     * `deployer`. If `deployer` is this contract's address, returns the same value as {computeAddress}.
     */
    function computeAddress(bytes32 salt, bytes32 bytecodeHash, address deployer) internal pure returns (address addr) {
        assembly ("memory-safe") {
            let ptr := mload(0x40) // Get free memory pointer

            // |                   | ↓ ptr ...  ↓ ptr + 0x0B (start) ...  ↓ ptr + 0x20 ...  ↓ ptr + 0x40 ...   |
            // |-------------------|---------------------------------------------------------------------------|
            // | bytecodeHash      |                                                        CCCCCCCCCCCCC...CC |
            // | salt              |                                      BBBBBBBBBBBBB...BB                   |
            // | deployer          | 000000...0000AAAAAAAAAAAAAAAAAAA...AA                                     |
            // | 0xFF              |            FF                                                             |
            // |-------------------|---------------------------------------------------------------------------|
            // | memory            | 000000...00FFAAAAAAAAAAAAAAAAAAA...AABBBBBBBBBBBBB...BBCCCCCCCCCCCCC...CC |
            // | keccak(start, 85) |            ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ |

            mstore(add(ptr, 0x40), bytecodeHash)
            mstore(add(ptr, 0x20), salt)
            mstore(ptr, deployer) // Right-aligned with 12 preceding garbage bytes
            let start := add(ptr, 0x0b) // The hashed data starts at the final garbage byte which we will set to 0xff
            mstore8(start, 0xff)
            addr := and(keccak256(start, 85), 0xffffffffffffffffffffffffffffffffffffffff)
        }
    }
}


// ===== FILE: _openzeppelin/contracts/utils/Errors.sol =====
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


// ===== FILE: _openzeppelin/contracts/utils/introspection/IERC165.sol =====
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


// ===== FILE: contracts-exposed/__/_openzeppelin/contracts/token/ERC20/ERC20.sol =====
// SPDX-License-Identifier: UNLICENSED

pragma solidity >=0.6.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/interfaces/draft-IERC6093.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/Context.sol";

contract $ERC20 is ERC20 {
    bytes32 public constant __hh_exposed_bytecode_marker = "hardhat-exposed";

    constructor(string memory name_, string memory symbol_) ERC20(name_, symbol_) payable {
    }

    function $_transfer(address from,address to,uint256 value) external payable {
        super._transfer(from,to,value);
    }

    function $_update(address from,address to,uint256 value) external payable {
        super._update(from,to,value);
    }

    function $_mint(address account,uint256 value) external payable {
        super._mint(account,value);
    }

    function $_burn(address account,uint256 value) external payable {
        super._burn(account,value);
    }

    function $_approve(address owner,address spender,uint256 value) external payable {
        super._approve(owner,spender,value);
    }

    function $_approve(address owner,address spender,uint256 value,bool emitEvent) external payable {
        super._approve(owner,spender,value,emitEvent);
    }

    function $_spendAllowance(address owner,address spender,uint256 value) external payable {
        super._spendAllowance(owner,spender,value);
    }

    function $_msgSender() external view returns (address ret0) {
        (ret0) = super._msgSender();
    }

    function $_msgData() external view returns (bytes memory ret0) {
        (ret0) = super._msgData();
    }

    function $_contextSuffixLength() external view returns (uint256 ret0) {
        (ret0) = super._contextSuffixLength();
    }

    receive() external payable {}
}


// ===== FILE: contracts-exposed/__/_openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol =====
// SPDX-License-Identifier: UNLICENSED

pragma solidity >=0.6.0;

import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/interfaces/IERC1363.sol";
import "@openzeppelin/contracts/interfaces/IERC20.sol";
import "@openzeppelin/contracts/interfaces/IERC165.sol";
import "@openzeppelin/contracts/utils/introspection/IERC165.sol";

contract $SafeERC20 {
    bytes32 public constant __hh_exposed_bytecode_marker = "hardhat-exposed";

    event return$trySafeTransfer(bool ret0);

    event return$trySafeTransferFrom(bool ret0);

    constructor() payable {
    }

    function $safeTransfer(IERC20 token,address to,uint256 value) external payable {
        SafeERC20.safeTransfer(token,to,value);
    }

    function $safeTransferFrom(IERC20 token,address from,address to,uint256 value) external payable {
        SafeERC20.safeTransferFrom(token,from,to,value);
    }

    function $trySafeTransfer(IERC20 token,address to,uint256 value) external payable returns (bool ret0) {
        (ret0) = SafeERC20.trySafeTransfer(token,to,value);
        emit return$trySafeTransfer(ret0);
    }

    function $trySafeTransferFrom(IERC20 token,address from,address to,uint256 value) external payable returns (bool ret0) {
        (ret0) = SafeERC20.trySafeTransferFrom(token,from,to,value);
        emit return$trySafeTransferFrom(ret0);
    }

    function $safeIncreaseAllowance(IERC20 token,address spender,uint256 value) external payable {
        SafeERC20.safeIncreaseAllowance(token,spender,value);
    }

    function $safeDecreaseAllowance(IERC20 token,address spender,uint256 requestedDecrease) external payable {
        SafeERC20.safeDecreaseAllowance(token,spender,requestedDecrease);
    }

    function $forceApprove(IERC20 token,address spender,uint256 value) external payable {
        SafeERC20.forceApprove(token,spender,value);
    }

    function $transferAndCallRelaxed(IERC1363 token,address to,uint256 value,bytes calldata data) external payable {
        SafeERC20.transferAndCallRelaxed(token,to,value,data);
    }

    function $transferFromAndCallRelaxed(IERC1363 token,address from,address to,uint256 value,bytes calldata data) external payable {
        SafeERC20.transferFromAndCallRelaxed(token,from,to,value,data);
    }

    function $approveAndCallRelaxed(IERC1363 token,address to,uint256 value,bytes calldata data) external payable {
        SafeERC20.approveAndCallRelaxed(token,to,value,data);
    }

    receive() external payable {}
}


// ===== FILE: contracts-exposed/__/_openzeppelin/contracts/utils/Context.sol =====
// SPDX-License-Identifier: UNLICENSED

pragma solidity >=0.6.0;

import "@openzeppelin/contracts/utils/Context.sol";

contract $Context is Context {
    bytes32 public constant __hh_exposed_bytecode_marker = "hardhat-exposed";

    constructor() payable {
    }

    function $_msgSender() external view returns (address ret0) {
        (ret0) = super._msgSender();
    }

    function $_msgData() external view returns (bytes memory ret0) {
        (ret0) = super._msgData();
    }

    function $_contextSuffixLength() external view returns (uint256 ret0) {
        (ret0) = super._contextSuffixLength();
    }

    receive() external payable {}
}


// ===== FILE: contracts-exposed/__/_openzeppelin/contracts/utils/Create2.sol =====
// SPDX-License-Identifier: UNLICENSED

pragma solidity >=0.6.0;

import "@openzeppelin/contracts/utils/Create2.sol";
import "@openzeppelin/contracts/utils/Errors.sol";

contract $Create2 {
    bytes32 public constant __hh_exposed_bytecode_marker = "hardhat-exposed";

    event return$deploy(address addr);

    constructor() payable {
    }

    function $deploy(uint256 amount,bytes32 salt,bytes calldata bytecode) external payable returns (address addr) {
        (addr) = Create2.deploy(amount,salt,bytecode);
        emit return$deploy(addr);
    }

    function $computeAddress(bytes32 salt,bytes32 bytecodeHash) external view returns (address ret0) {
        (ret0) = Create2.computeAddress(salt,bytecodeHash);
    }

    function $computeAddress(bytes32 salt,bytes32 bytecodeHash,address deployer) external pure returns (address addr) {
        (addr) = Create2.computeAddress(salt,bytecodeHash,deployer);
    }

    receive() external payable {}
}


// ===== FILE: contracts-exposed/__/_openzeppelin/contracts/utils/Errors.sol =====
// SPDX-License-Identifier: UNLICENSED

pragma solidity >=0.6.0;

import "@openzeppelin/contracts/utils/Errors.sol";

contract $Errors {
    bytes32 public constant __hh_exposed_bytecode_marker = "hardhat-exposed";

    constructor() payable {
    }

    receive() external payable {}
}


// ===== FILE: contracts-exposed/mocks/MockERC20.sol =====
// SPDX-License-Identifier: UNLICENSED

pragma solidity >=0.6.0;

import "../../contracts/mocks/MockERC20.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/interfaces/draft-IERC6093.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/Context.sol";

contract $MockERC20 is MockERC20 {
    bytes32 public constant __hh_exposed_bytecode_marker = "hardhat-exposed";

    constructor(string memory name, string memory symbol, uint8 _dec) MockERC20(name, symbol, _dec) payable {
    }

    function $_decimals() external view returns (uint8) {
        return _decimals;
    }

    function $_transfer(address from,address to,uint256 value) external payable {
        super._transfer(from,to,value);
    }

    function $_update(address from,address to,uint256 value) external payable {
        super._update(from,to,value);
    }

    function $_mint(address account,uint256 value) external payable {
        super._mint(account,value);
    }

    function $_burn(address account,uint256 value) external payable {
        super._burn(account,value);
    }

    function $_approve(address owner,address spender,uint256 value) external payable {
        super._approve(owner,spender,value);
    }

    function $_approve(address owner,address spender,uint256 value,bool emitEvent) external payable {
        super._approve(owner,spender,value,emitEvent);
    }

    function $_spendAllowance(address owner,address spender,uint256 value) external payable {
        super._spendAllowance(owner,spender,value);
    }

    function $_msgSender() external view returns (address ret0) {
        (ret0) = super._msgSender();
    }

    function $_msgData() external view returns (bytes memory ret0) {
        (ret0) = super._msgData();
    }

    function $_contextSuffixLength() external view returns (uint256 ret0) {
        (ret0) = super._contextSuffixLength();
    }

    receive() external payable {}
}


// ===== FILE: contracts-exposed/mocks/SimplePriceFeed.sol =====
// SPDX-License-Identifier: UNLICENSED

pragma solidity >=0.6.0;

import "../../contracts/mocks/SimplePriceFeed.sol";
import "../../contracts/interfaces/AggregatorV3Interface.sol";

contract $SimplePriceFeed is SimplePriceFeed {
    bytes32 public constant __hh_exposed_bytecode_marker = "hardhat-exposed";

    constructor(int256 answer_, uint8 decimals_) SimplePriceFeed(answer_, decimals_) payable {
    }

    function $roundId() external view returns (uint80) {
        return roundId;
    }

    function $answer() external view returns (int256) {
        return answer;
    }

    function $startedAt() external view returns (uint256) {
        return startedAt;
    }

    function $updatedAt() external view returns (uint256) {
        return updatedAt;
    }

    function $answeredInRound() external view returns (uint80) {
        return answeredInRound;
    }

    receive() external payable {}
}


// ===== FILE: contracts-exposed/oracles/MultiplicativePriceFeed.sol =====
// SPDX-License-Identifier: UNLICENSED

pragma solidity >=0.6.0;

import "../../contracts/oracles/MultiplicativePriceFeed.sol";
import "../../contracts/interfaces/IPriceFeed.sol";

contract $MultiplicativePriceFeed is MultiplicativePriceFeed {
    bytes32 public constant __hh_exposed_bytecode_marker = "hardhat-exposed";

    constructor(address priceFeedA_, address priceFeedB_, uint8 decimals_, string memory description_) MultiplicativePriceFeed(priceFeedA_, priceFeedB_, decimals_, description_) payable {
    }

    function $signed256(uint256 n) external pure returns (int256 ret0) {
        (ret0) = super.signed256(n);
    }

    receive() external payable {}
}


// ===== FILE: contracts-exposed/oracles/ReverseMultiplicativePriceFeed.sol =====
// SPDX-License-Identifier: UNLICENSED

pragma solidity >=0.6.0;

import "../../contracts/oracles/ReverseMultiplicativePriceFeed.sol";
import "../../contracts/interfaces/IPriceFeed.sol";

contract $ReverseMultiplicativePriceFeed is ReverseMultiplicativePriceFeed {
    bytes32 public constant __hh_exposed_bytecode_marker = "hardhat-exposed";

    constructor(address priceFeedA_, address priceFeedB_, uint8 decimals_, string memory description_) ReverseMultiplicativePriceFeed(priceFeedA_, priceFeedB_, decimals_, description_) payable {
    }

    function $signed256(uint256 n) external pure returns (int256 ret0) {
        (ret0) = super.signed256(n);
    }

    receive() external payable {}
}


// ===== FILE: contracts-exposed/Streamer.sol =====
// SPDX-License-Identifier: UNLICENSED

pragma solidity >=0.6.0;

import "../contracts/Streamer.sol";
import "../contracts/interfaces/IStreamer.sol";
import "../contracts/interfaces/AggregatorV3Interface.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/interfaces/IERC1363.sol";
import "@openzeppelin/contracts/interfaces/IERC20.sol";
import "@openzeppelin/contracts/interfaces/IERC165.sol";
import "@openzeppelin/contracts/utils/introspection/IERC165.sol";

contract $Streamer is Streamer {
    bytes32 public constant __hh_exposed_bytecode_marker = "hardhat-exposed";

    constructor(IERC20 _streamingAsset, AggregatorV3Interface _streamingAssetOracle, AggregatorV3Interface _nativeAssetOracle, address _returnAddress, address _streamCreator, address _recipient, uint8 _streamingAssetDecimals, uint8 _nativeAssetDecimals, uint256 _nativeAssetStreamingAmount, uint256 _slippage, uint256 _claimCooldown, uint256 _sweepCooldown, uint256 _streamDuration, uint256 _minimumNoticePeriod) Streamer(_streamingAsset, _streamingAssetOracle, _nativeAssetOracle, _returnAddress, _streamCreator, _recipient, _streamingAssetDecimals, _nativeAssetDecimals, _nativeAssetStreamingAmount, _slippage, _claimCooldown, _sweepCooldown, _streamDuration, _minimumNoticePeriod) payable {
    }

    function $onlyStreamCreator() external payable onlyStreamCreator() {}

    function $scaleAmount(uint256 amount,uint256 fromDecimals,uint256 toDecimals) external pure returns (uint256 ret0) {
        (ret0) = super.scaleAmount(amount,fromDecimals,toDecimals);
    }

    receive() external payable {}
}


// ===== FILE: contracts-exposed/StreamerFactory.sol =====
// SPDX-License-Identifier: UNLICENSED

pragma solidity >=0.6.0;

import "../contracts/StreamerFactory.sol";
import "../contracts/interfaces/IStreamerFactory.sol";
import "../contracts/interfaces/AggregatorV3Interface.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import "@openzeppelin/contracts/utils/Create2.sol";
import "../contracts/Streamer.sol";
import "@openzeppelin/contracts/utils/Errors.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "../contracts/interfaces/IStreamer.sol";
import "@openzeppelin/contracts/interfaces/IERC1363.sol";
import "@openzeppelin/contracts/interfaces/IERC20.sol";
import "@openzeppelin/contracts/interfaces/IERC165.sol";
import "@openzeppelin/contracts/utils/introspection/IERC165.sol";

contract $StreamerFactory is StreamerFactory {
    bytes32 public constant __hh_exposed_bytecode_marker = "hardhat-exposed";

    constructor() payable {
    }

    receive() external payable {}
}


// ===== FILE: contracts/interfaces/AggregatorV3Interface.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

interface AggregatorV3Interface {
    function decimals() external view returns (uint8);

    function description() external view returns (string memory);

    function version() external view returns (uint256);

    // getRoundData and latestRoundData should both raise "No data present"
    // if they do not have data to report, instead of returning unset values
    // which could be misinterpreted as actual reported values.
    function getRoundData(
        uint80 _roundId
    )
        external
        view
        returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound);

    function latestRoundData()
        external
        view
        returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound);
}


// ===== FILE: contracts/interfaces/IComptrollerV2.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

interface IComptrollerV2 {
    function _grantComp(address recipient, uint256 amount) external;
}


// ===== FILE: contracts/interfaces/IPriceFeed.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

/**
 * @dev Interface for price feeds used by Comet
 * Note This is Chainlink's AggregatorV3Interface, but without the `getRoundData` function.
 */
interface IPriceFeed {
    function decimals() external view returns (uint8);

    function description() external view returns (string memory);

    function version() external view returns (uint256);

    function latestRoundData()
        external
        view
        returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound);
}


// ===== FILE: contracts/interfaces/IStreamer.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

import { IERC20 } from "@openzeppelin/contracts/token/ERC20/IERC20.sol";

enum StreamState {
    NOT_INITIALIZED,
    STARTED,
    SHORTENED,
    FINISHED
}

interface IStreamer {
    event Initialized();
    event Claimed(uint256 streamingAssetAmount, uint256 nativeAssetAmount);
    event Terminated(uint256 terminationTimestamp);
    event Swept(uint256 amount);
    event Rescued(address token, uint256 balance);
    event InsufficientAssetBalance(uint256 balanceRequired, uint256 balance);

    error ZeroAmount();
    error NotReceiver();
    error NotStreamCreator();
    error CantRescueStreamingAsset();
    error ZeroAddress();
    error SlippageExceedsScaleFactor();
    error InvalidPrice();
    error NotInitialized();
    error NotEnoughBalance(uint256 balance, uint256 streamingAmount);
    error StreamNotFinished();
    error AlreadyInitialized();
    error DurationTooShort();
    error TerminationIsAfterStream(uint256 terminationTimestamp);
    error CreatorCannotSweepYet();
    error SweepCooldownNotPassed();
    error AlreadyTerminated();
    error NoticePeriodExceedsStreamDuration();
    error DecimalsNotInBounds();
    error StreamingAmountTooLow();

    function initialize() external;

    function claim() external;

    function sweepRemaining() external;

    function terminateStream(uint256 _terminationTimestamp) external;

    function rescueToken(IERC20 token) external;

    function getNativeAssetAmountOwed() external view returns (uint256);

    function calculateStreamingAssetAmount(uint256 nativeAssetAmount) external view returns (uint256);

    function calculateNativeAssetAmount(uint256 streamingAssetAmount) external view returns (uint256);
}


// ===== FILE: contracts/interfaces/IStreamerFactory.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

import { AggregatorV3Interface } from "./AggregatorV3Interface.sol";

interface IStreamerFactory {
    event StreamerDeployed(address newContract, bytes constructorParams);

    error AssetsMatch();

    function deployStreamer(
        address _streamingAsset,
        address _nativeAsset,
        AggregatorV3Interface _streamingAssetOracle,
        AggregatorV3Interface _nativeAssetOracle,
        address _returnAddress,
        address _streamCreator,
        address _recipient,
        uint256 _nativeAssetStreamingAmount,
        uint256 _slippage,
        uint256 _sweepCooldown,
        uint256 _finishCooldown,
        uint256 _streamDuration,
        uint256 _minimumNoticePeriod
    ) external returns (address);
}


// ===== FILE: contracts/mocks/MockERC20.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

import { ERC20 } from "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract MockERC20 is ERC20 {
    uint8 internal immutable _decimals;

    constructor(string memory name, string memory symbol, uint8 _dec) ERC20(name, symbol) {
        //require(_dec <= 30, "Invalid Decimals!");
        _decimals = _dec;
    }

    function decimals() public view override returns (uint8) {
        return _decimals;
    }

    function mint(address to, uint256 amount) external {
        _mint(to, amount);
    }
}


// ===== FILE: contracts/mocks/SimplePriceFeed.sol =====
// SPDX-License-Identifier: BUSL-1.1
pragma solidity ^0.8.29;

import "../interfaces/AggregatorV3Interface.sol";

contract SimplePriceFeed is AggregatorV3Interface {
    string public constant override description = "Mock Chainlink price aggregator";

    uint public constant override version = 1;

    uint8 public immutable override decimals;

    uint80 internal roundId;
    int256 internal answer;
    uint256 internal startedAt;
    uint256 internal updatedAt;
    uint80 internal answeredInRound;

    constructor(int answer_, uint8 decimals_) {
        answer = answer_;
        decimals = decimals_;
    }

    function setPrice(int256 answer_) public {
        answer = answer_;
    }

    function setRoundData(
        uint80 roundId_,
        int256 answer_,
        uint256 startedAt_,
        uint256 updatedAt_,
        uint80 answeredInRound_
    ) public {
        roundId = roundId_;
        answer = answer_;
        startedAt = startedAt_;
        updatedAt = updatedAt_;
        answeredInRound = answeredInRound_;
    }

    function getRoundData(uint80 roundId_) external view override returns (uint80, int256, uint256, uint256, uint80) {
        return (roundId_, answer, startedAt, updatedAt, answeredInRound);
    }

    function latestRoundData() external view override returns (uint80, int256, uint256, uint256, uint80) {
        return (roundId, answer, startedAt, updatedAt, answeredInRound);
    }
}


// ===== FILE: contracts/oracles/MultiplicativePriceFeed.sol =====
// SPDX-License-Identifier: BUSL-1.1
pragma solidity ^0.8.29;

import "../interfaces/AggregatorV3Interface.sol";
import "../interfaces/IPriceFeed.sol";

/**
 * @title Multiplicative price feed
 * @notice A custom price feed that multiplies the prices from two price feeds and returns the result
 * @author Compound
 */
contract MultiplicativePriceFeed is IPriceFeed {
    /** Custom errors **/
    error BadDecimals();
    error InvalidInt256();

    /// @notice Version of the price feed
    uint public constant VERSION = 1;

    /// @notice Description of the price feed
    string public override description;

    /// @notice Number of decimals for returned prices
    uint8 public immutable override decimals;

    /// @notice Chainlink price feed A
    address public immutable priceFeedA;

    /// @notice Chainlink price feed B
    address public immutable priceFeedB;

    /// @notice Combined scale of the two underlying Chainlink price feeds
    int public immutable combinedScale;

    /// @notice Scale of this price feed
    int public immutable priceFeedScale;

    /**
     * @notice Construct a new multiplicative price feed
     * @param priceFeedA_ The address of the first price feed to fetch prices from
     * @param priceFeedB_ The address of the second price feed to fetch prices from
     * @param decimals_ The number of decimals for the returned prices
     * @param description_ The description of the price feed
     **/
    constructor(address priceFeedA_, address priceFeedB_, uint8 decimals_, string memory description_) {
        priceFeedA = priceFeedA_;
        priceFeedB = priceFeedB_;
        uint8 priceFeedADecimals = AggregatorV3Interface(priceFeedA_).decimals();
        uint8 priceFeedBDecimals = AggregatorV3Interface(priceFeedB_).decimals();
        combinedScale = signed256(10 ** (priceFeedADecimals + priceFeedBDecimals));

        if (decimals_ > 18) revert BadDecimals();
        decimals = decimals_;
        description = description_;
        priceFeedScale = int256(10 ** decimals);
    }

    /**
     * @notice Calculates the latest round data using data from the two price feeds
     * @return roundId Round id from price feed B
     * @return answer Latest price
     * @return startedAt Timestamp when the round was started; passed on from price feed B
     * @return updatedAt Timestamp when the round was last updated; passed on from price feed B
     * @return answeredInRound Round id in which the answer was computed; passed on from price feed B
     * @dev Note: Only the `answer` really matters for downstream contracts that use this price feed (e.g. Comet)
     **/
    function latestRoundData() external view override returns (uint80, int256, uint256, uint256, uint80) {
        (, int256 priceA, , , ) = AggregatorV3Interface(priceFeedA).latestRoundData();
        (
            uint80 roundId_,
            int256 priceB,
            uint256 startedAt_,
            uint256 updatedAt_,
            uint80 answeredInRound_
        ) = AggregatorV3Interface(priceFeedB).latestRoundData();

        if (priceA <= 0 || priceB <= 0) return (roundId_, 0, startedAt_, updatedAt_, answeredInRound_);

        int256 price = (priceA * priceB * priceFeedScale) / combinedScale;
        return (roundId_, price, startedAt_, updatedAt_, answeredInRound_);
    }

    function signed256(uint256 n) internal pure returns (int256) {
        if (n > uint256(type(int256).max)) revert InvalidInt256();
        return int256(n);
    }

    /**
     * @notice Price for the latest round
     * @return The version of the price feed contract
     **/
    function version() external pure returns (uint256) {
        return VERSION;
    }
}


// ===== FILE: contracts/oracles/ReverseMultiplicativePriceFeed.sol =====
// SPDX-License-Identifier: BUSL-1.1
pragma solidity ^0.8.29;

import "../interfaces/AggregatorV3Interface.sol";
import "../interfaces/IPriceFeed.sol";

/**
 * @title Reverse multiplicative price feed
 * @notice A custom price feed that multiplies the price from one price feed and the inverse price from another price feed and returns the result
 * @dev for example if we need tokenX to eth, but there is only tokenX to usd, we can use this price feed to get tokenX to eth: tokenX to usd * reversed(eth to usd)
 * @author Compound
 */
contract ReverseMultiplicativePriceFeed is IPriceFeed {
    /** Custom errors **/
    error BadDecimals();
    error InvalidInt256();

    /// @notice Version of the price feed
    uint public constant VERSION = 1;

    /// @notice Description of the price feed
    string public override description;

    /// @notice Number of decimals for returned prices
    uint8 public immutable override decimals;

    /// @notice Chainlink price feed A
    address public immutable priceFeedA;

    /// @notice Chainlink price feed B
    address public immutable priceFeedB;

    /// @notice Price feed A scale
    int public immutable priceFeedAScale;

    /// @notice Price feed B scale
    int public immutable priceFeedBScale;

    /// @notice Scale of this price feed
    int public immutable priceFeedScale;

    /**
     * @notice Construct a new reverse multiplicative price feed
     * @param priceFeedA_ The address of the first price feed to fetch prices from
     * @param priceFeedB_ The address of the second price feed to fetch prices from that should be reversed
     * @param decimals_ The number of decimals for the returned prices
     * @param description_ The description of the price feed
     **/
    constructor(address priceFeedA_, address priceFeedB_, uint8 decimals_, string memory description_) {
        priceFeedA = priceFeedA_;
        priceFeedB = priceFeedB_;
        uint8 priceFeedADecimals = AggregatorV3Interface(priceFeedA_).decimals();
        uint8 priceFeedBDecimals = AggregatorV3Interface(priceFeedB_).decimals();
        priceFeedAScale = signed256(10 ** (priceFeedADecimals));
        priceFeedBScale = signed256(10 ** (priceFeedBDecimals));

        if (decimals_ > 18) revert BadDecimals();
        decimals = decimals_;
        description = description_;
        priceFeedScale = int256(10 ** decimals);
    }

    /**
     * @notice Calculates the latest round data using data from the two price feeds
     * @return roundId Round id from price feed B
     * @return answer Latest price
     * @return startedAt Timestamp when the round was started; passed on from price feed B
     * @return updatedAt Timestamp when the round was last updated; passed on from price feed B
     * @return answeredInRound Round id in which the answer was computed; passed on from price feed B
     * @dev Note: Only the `answer` really matters for downstream contracts that use this price feed (e.g. Comet)
     **/
    function latestRoundData() external view override returns (uint80, int256, uint256, uint256, uint80) {
        (, int256 priceA, , , ) = AggregatorV3Interface(priceFeedA).latestRoundData();
        (
            uint80 roundId_,
            int256 priceB,
            uint256 startedAt_,
            uint256 updatedAt_,
            uint80 answeredInRound_
        ) = AggregatorV3Interface(priceFeedB).latestRoundData();

        if (priceA <= 0 || priceB <= 0) return (roundId_, 0, startedAt_, updatedAt_, answeredInRound_);

        // int256 price = priceA * (priceFeedBScale/priceB) * priceFeedScale / priceFeedAScale;
        int256 price = (priceA * priceFeedBScale * priceFeedScale) / priceB / priceFeedAScale;
        return (roundId_, price, startedAt_, updatedAt_, answeredInRound_);
    }

    function signed256(uint256 n) internal pure returns (int256) {
        if (n > uint256(type(int256).max)) revert InvalidInt256();
        return int256(n);
    }

    /**
     * @notice Price for the latest round
     * @return The version of the price feed contract
     **/
    function version() external pure returns (uint256) {
        return VERSION;
    }
}


// ===== FILE: contracts/Streamer.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

// Developed with @openzeppelin/contracts v5.3.0
import { AggregatorV3Interface } from "./interfaces/AggregatorV3Interface.sol";
import { IERC20 } from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import { SafeERC20 } from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import { StreamState, IStreamer } from "./interfaces/IStreamer.sol";

/** @title Streamer
 * @author WOOF! Software
 * @custom:security-contact dmitriy@woof.software
 * @notice This contract streams a certain amount of native asset in a form of streaming asset to the recipient over a specified streaming duration.
 * - The contract is designed to work with a pair of Chainlink oracles: Native Asset / USD and Streaming Asset / USD. However, can support any oracle which supports AggregatorV3Interface.
 * - Streaming asset is accrued linearly over a streaming duration, unlocking a portion of Streaming asset each second. Recipient can claim any time during and after the stream.
 * - Stream Creator is able to:
 *  1. rescue any ERC-20 token stuck in contract except of streaming asset.
 *  2. terminate the stream until the stream end. In this case, the distribution of streaming asset will continue till the termination timestamp.
 *  3. sweep remaining streaming asset tokens after stream end or termination timestamp (in case the stream is terminated).
 * - The streaming amount is specified in the native asset units. During the claiming, accrued native asset amount is calculated into streaming asset.
 * - All the tokens transferred via sweepRemaining or rescueToken are sent to the returnAddress.
 * - Anyone is able to call claim or sweepRemaining after a specified duration. Assets will still be transferred to the recipient and returnAddress accordingly.
 */
contract Streamer is IStreamer {
    using SafeERC20 for IERC20;

    /// @notice The denominator for slippage calculation. Equals 100%.
    uint256 public constant SLIPPAGE_SCALE = 1e8;
    /// @notice Minimal required duration for all duration parameters.
    uint256 public constant MIN_DURATION = 1 days;
    /// @notice Minimal number of decimals allowed for tokens and price feeds.
    uint8 public constant MIN_DECIMALS = 6;
    /// @notice Number of decimals used to scale prices.
    uint8 public constant SCALE_DECIMALS = 18;

    /// @notice The address of asset used for distribution.
    IERC20 public immutable streamingAsset;
    /// @notice The address of price feed oracle for Streaming asset. Must return the price in USD.
    AggregatorV3Interface public immutable streamingAssetOracle;
    /// @notice The address of price feed oracle for Native asset. Must return the price in USD.
    AggregatorV3Interface public immutable nativeAssetOracle;
    /// @notice The address which receives tokens during the execution of sweepRemaining and rescueToken functions.
    address public immutable returnAddress;
    /// @notice The owner of the stream.
    address public immutable streamCreator;
    /// @notice The recipient of streaming asset.
    address public immutable recipient;
    /// @notice Amount of asset to be distributed. Specified in the Native asset units.
    uint256 public immutable nativeAssetStreamingAmount;
    /// @notice A percentage used to reduce the price of streaming asset to account for price fluctuations.
    uint256 public immutable slippage;
    /// @notice A period of time since last claim timestamp after which anyone can call claim.
    uint256 public immutable claimCooldown;
    /// @notice A period of time since the end of stream after which anyone can call sweepRemaining.
    uint256 public immutable sweepCooldown;
    /// @notice A period of time since the initialization of the stream during which asset is streamed.
    uint256 public immutable streamDuration;
    /// @notice A minimal period of time during which Streaming asset must continue to accrue after termination is called.
    uint256 public immutable minimumNoticePeriod;
    /// @notice Decimals of Streaming asset.
    uint8 public immutable streamingAssetDecimals;
    /// @notice Decimals of Native asset.
    uint8 public immutable nativeAssetDecimals;
    /// @notice Decimals of the price returned by the Streaming Asset Oracle.
    uint8 public immutable streamingAssetOracleDecimals;
    /// @notice Decimals of the price returned by the Native Asset Oracle.
    uint8 public immutable nativeAssetOracleDecimals;
    /// @notice The start of the stream. Set during initialization of the stream.
    uint256 public startTimestamp;
    /// @notice The timestamp of the latest claim call.
    uint256 public lastClaimTimestamp;
    /// @notice The timestamp till which tokens continue to accrue. Set during the terminateStream call.
    uint256 public terminationTimestamp;
    /// @notice Amount of Native asset already distributed.
    uint256 public nativeAssetSuppliedAmount;
    /// @notice Total amount of claimed Streaming asset.
    uint256 public streamingAssetClaimedAmount;
    /// @notice The state which indicated if the stream is not initialized, ongoing or terminated.
    StreamState private state;

    modifier onlyStreamCreator() {
        if (msg.sender != streamCreator) revert NotStreamCreator();
        _;
    }

    /// @dev Decimals for tokens and price feeds should be between 6 and 18 to ensure proper calculations.
    /// @dev Streaming asset should not be a token with multiple addresses to ensure the correct flow of the stream.
    /// USD value of `_nativeAssetStreamingAmount` must be equal to at least $1.
    constructor(
        IERC20 _streamingAsset,
        AggregatorV3Interface _streamingAssetOracle,
        AggregatorV3Interface _nativeAssetOracle,
        address _returnAddress,
        address _streamCreator,
        address _recipient,
        uint8 _streamingAssetDecimals,
        uint8 _nativeAssetDecimals,
        uint256 _nativeAssetStreamingAmount,
        uint256 _slippage,
        uint256 _claimCooldown,
        uint256 _sweepCooldown,
        uint256 _streamDuration,
        uint256 _minimumNoticePeriod
    ) {
        if (_recipient == address(0)) revert ZeroAddress();
        if (_streamCreator == address(0)) revert ZeroAddress();
        if (_returnAddress == address(0)) revert ZeroAddress();
        if (address(_streamingAsset) == address(0)) revert ZeroAddress();
        if (_nativeAssetStreamingAmount == 0) revert ZeroAmount();
        if (_slippage > SLIPPAGE_SCALE) revert SlippageExceedsScaleFactor();
        if (_claimCooldown < MIN_DURATION) revert DurationTooShort();
        if (_sweepCooldown < MIN_DURATION) revert DurationTooShort();
        if (_streamDuration < MIN_DURATION) revert DurationTooShort();
        if (_minimumNoticePeriod < MIN_DURATION) revert DurationTooShort();
        if (_minimumNoticePeriod > _streamDuration) revert NoticePeriodExceedsStreamDuration();
        streamingAssetOracleDecimals = _streamingAssetOracle.decimals();
        nativeAssetOracleDecimals = _nativeAssetOracle.decimals();
        if (
            _streamingAssetDecimals < MIN_DECIMALS ||
            _nativeAssetDecimals < MIN_DECIMALS ||
            streamingAssetOracleDecimals < MIN_DECIMALS ||
            nativeAssetOracleDecimals < MIN_DECIMALS
        ) revert DecimalsNotInBounds();
        (, int256 nativeAssetPrice, , , ) = _nativeAssetOracle.latestRoundData();
        if (nativeAssetPrice <= 0) revert InvalidPrice();
        if (
            (_nativeAssetStreamingAmount * uint256(nativeAssetPrice)) / 10 ** nativeAssetOracleDecimals <
            10 ** _nativeAssetDecimals
        ) revert StreamingAmountTooLow();

        streamingAsset = _streamingAsset;
        streamingAssetOracle = _streamingAssetOracle;
        nativeAssetOracle = _nativeAssetOracle;
        returnAddress = _returnAddress;
        streamCreator = _streamCreator;
        recipient = _recipient;
        streamingAssetDecimals = _streamingAssetDecimals;
        nativeAssetDecimals = _nativeAssetDecimals;
        nativeAssetStreamingAmount = _nativeAssetStreamingAmount;
        slippage = _slippage;
        claimCooldown = _claimCooldown;
        sweepCooldown = _sweepCooldown;
        streamDuration = _streamDuration;
        minimumNoticePeriod = _minimumNoticePeriod;
    }

    /** @notice Initializes the stream by setting start timestamp and validating that the contract has enough Streaming asset.
     * @dev Streaming asset must be transferred to the contract's balance before function is called.
     * @dev It is recommended to send a sufficient amount of Streaming asset in order to ensure the correct work of the Streamer.
     * The extra amount depends on the volatility of assets. In general, we recommend sending extra 10% of the necessary Streaming asset amount.
     * @dev Use the function `calculateStreamingAssetAmount` to determine the amount of Streaming asset to transfer.
     */
    function initialize() external {
        if (state != StreamState.NOT_INITIALIZED) revert AlreadyInitialized();
        startTimestamp = block.timestamp;
        lastClaimTimestamp = block.timestamp;
        state = StreamState.STARTED;

        uint256 balance = streamingAsset.balanceOf(address(this));
        if (calculateNativeAssetAmount(balance) < nativeAssetStreamingAmount)
            revert NotEnoughBalance(balance, nativeAssetStreamingAmount);

        emit Initialized();
    }

    /** @notice Claims the accrued amount of Streaming asset to the recipient's address.
     * @dev The stream must be initialized.
     * @dev Can be called by the recipient or anyone after claim cooldown has passed since the last claim timestamp.
     * @dev In case the contract doesn't have enough Streaming asset on its balance, the whole balance will be sent. The stream owner will have to replenish
     * the balance in order to resume the stream.
     */
    function claim() external {
        if (state == StreamState.NOT_INITIALIZED) revert NotInitialized();
        if (msg.sender != recipient && block.timestamp < lastClaimTimestamp + claimCooldown) revert NotReceiver();

        uint256 owed = getNativeAssetAmountOwed();
        if (owed == 0) revert ZeroAmount();

        uint256 streamingAssetAmount = calculateStreamingAssetAmount(owed);
        if (streamingAssetAmount == 0) revert ZeroAmount();

        uint256 balance = streamingAsset.balanceOf(address(this));
        if (balance < streamingAssetAmount) {
            emit InsufficientAssetBalance(streamingAssetAmount, balance);
            streamingAssetAmount = balance;
            owed = calculateNativeAssetAmount(balance);
        }

        lastClaimTimestamp = block.timestamp;
        nativeAssetSuppliedAmount += owed;
        streamingAssetClaimedAmount += streamingAssetAmount;

        streamingAsset.safeTransfer(recipient, streamingAssetAmount);
        emit Claimed(streamingAssetAmount, owed);
    }

    /// @notice Terminates the stream, stopping the distribution after the termination timestamp.
    /// @param _terminationTimestamp The timestamp after which the stream is stopped. Must be longer than `block.timestamp + minimumNoticePeriod` and less than the end of stream.
    /// If the parameter is equal to 0, the termination timestamp will be set as `block.timestamp + minimumNoticePeriod`.
    function terminateStream(uint256 _terminationTimestamp) external onlyStreamCreator {
        if (state == StreamState.SHORTENED) revert AlreadyTerminated();
        if (_terminationTimestamp == 0) {
            terminationTimestamp = block.timestamp + minimumNoticePeriod;
        } else {
            if (_terminationTimestamp < block.timestamp + minimumNoticePeriod) revert DurationTooShort();
            terminationTimestamp = _terminationTimestamp;
        }

        if (terminationTimestamp > startTimestamp + streamDuration)
            revert TerminationIsAfterStream(_terminationTimestamp);
        state = StreamState.SHORTENED;
        emit Terminated(terminationTimestamp);
    }

    /** @notice Allows to sweep all the Streaming asset tokens from the Streamer's balance.
     * @dev Can be called by Stream Creator before initialization without any additional conditions.
     * @dev After the end of stream (Either after stream duration or after termination timestamp if termination was called), can be called
     * by Stream Creator or anyone after sweep cooldown has passed.
     */
    function sweepRemaining() external {
        if (state == StreamState.NOT_INITIALIZED) {
            if (msg.sender != streamCreator) {
                revert NotStreamCreator();
            }
        } else {
            uint256 streamEnd = getStreamEnd();

            if (msg.sender == streamCreator) {
                if (block.timestamp <= streamEnd) {
                    revert CreatorCannotSweepYet();
                }
            } else if (block.timestamp <= streamEnd + sweepCooldown) {
                revert SweepCooldownNotPassed();
            }
        }
        uint256 remainingBalance = streamingAsset.balanceOf(address(this));

        streamingAsset.safeTransfer(returnAddress, remainingBalance);
        emit Swept(remainingBalance);
    }

    /** @notice Allows to transfer any ERC-20 token except the Streaming asset from the Streamer's balance.
     * @param token Address of ERC-20 token to transfer.
     * @dev Can only be called by Stream Creator.
     */
    function rescueToken(IERC20 token) external onlyStreamCreator {
        if (token == streamingAsset) revert CantRescueStreamingAsset();
        uint256 balance = token.balanceOf(address(this));
        token.safeTransfer(returnAddress, balance);
        emit Rescued(address(token), balance);
    }

    /// @notice Calculates the amount of asset accrued since the last claiming
    /// @return Amount of accrued asset in Native asset units.
    function getNativeAssetAmountOwed() public view returns (uint256) {
        if (nativeAssetSuppliedAmount >= nativeAssetStreamingAmount) {
            return 0;
        }
        uint256 streamEnd = getStreamEnd();
        // Validate if stream is properly initialized
        if (streamEnd == 0) return 0;
        uint256 totalOwed;

        if (block.timestamp < streamEnd) {
            uint256 elapsed = block.timestamp - startTimestamp;
            totalOwed = (nativeAssetStreamingAmount * elapsed) / streamDuration;
        } else {
            // If Stream is terminated, calculate amount accrued before termination timestamp
            if (state == StreamState.SHORTENED)
                totalOwed = (nativeAssetStreamingAmount * (streamEnd - startTimestamp)) / streamDuration;
            else totalOwed = nativeAssetStreamingAmount;
        }
        return totalOwed - nativeAssetSuppliedAmount;
    }

    /** @notice Calculates the amount of Streaming asset based on the specified Native asset amount.
     * @param nativeAssetAmount The amount of Native asset to be converted to Streaming asset.
     * @dev Used in `claim` to calculate the amount Native asset owed in Streaming asset.
     * @dev The price of streaming asset is reduced by the slippage to account for price fluctuations.
     * @return Amount of Streaming asset.
     */
    function calculateStreamingAssetAmount(uint256 nativeAssetAmount) public view returns (uint256) {
        (, int256 streamingAssetPrice, , , ) = streamingAssetOracle.latestRoundData();
        if (streamingAssetPrice <= 0) revert InvalidPrice();

        (, int256 nativeAssetPrice, , , ) = nativeAssetOracle.latestRoundData();
        if (nativeAssetPrice <= 0) revert InvalidPrice();

        uint256 streamingAssetPriceScaled = (scaleAmount(
            uint256(streamingAssetPrice),
            streamingAssetOracleDecimals,
            SCALE_DECIMALS
        ) * (SLIPPAGE_SCALE - slippage)) / SLIPPAGE_SCALE;
        // Scale native asset price to streaming asset decimals for calculations
        uint256 nativeAssetPriceScaled = scaleAmount(
            uint256(nativeAssetPrice),
            nativeAssetOracleDecimals,
            SCALE_DECIMALS
        );
        uint256 amountInStreamingAsset = (scaleAmount(nativeAssetAmount, nativeAssetDecimals, SCALE_DECIMALS) *
            nativeAssetPriceScaled) / streamingAssetPriceScaled;

        return scaleAmount(amountInStreamingAsset, SCALE_DECIMALS, streamingAssetDecimals);
    }

    /** @notice Calculates the amount of Native asset based on the specified Streaming asset amount.
     * @param streamingAssetAmount The amount of Streaming asset to be converted to Native asset.
     * @dev Used in `initialize` to validate if the Streamer has enough Streaming asset to begin stream.
     * @dev Used in `claim` to calculate how much the remaining balance of Streaming asset is equal to the Native Asset
     * (For cases where the Streamer doesn't have enough Streaming asset to distribute).
     * @return Amount of Native asset.
     */
    function calculateNativeAssetAmount(uint256 streamingAssetAmount) public view returns (uint256) {
        (, int256 streamingAssetPrice, , , ) = streamingAssetOracle.latestRoundData();
        if (streamingAssetPrice <= 0) revert InvalidPrice();

        (, int256 nativeAssetPrice, , , ) = nativeAssetOracle.latestRoundData();
        if (nativeAssetPrice <= 0) revert InvalidPrice();

        // Streaming asset price is reduced by slippage to account for price fluctuations
        uint256 streamingAssetPriceScaled = (scaleAmount(
            uint256(streamingAssetPrice),
            streamingAssetOracleDecimals,
            SCALE_DECIMALS
        ) * (SLIPPAGE_SCALE - slippage)) / SLIPPAGE_SCALE;
        // Scale native asset price to streaming asset decimals for calculations
        uint256 nativeAssetPriceScaled = scaleAmount(
            uint256(nativeAssetPrice),
            nativeAssetOracleDecimals,
            SCALE_DECIMALS
        );
        uint256 amountInNativeAsset = (scaleAmount(streamingAssetAmount, streamingAssetDecimals, SCALE_DECIMALS) *
            streamingAssetPriceScaled) / nativeAssetPriceScaled;

        return scaleAmount(amountInNativeAsset, SCALE_DECIMALS, nativeAssetDecimals);
    }

    /// @dev Returns a correct end of the stream once the stream is initialized.
    /// @return Timestamp representing the end of the stream.
    function getStreamEnd() public view returns (uint256) {
        if (state == StreamState.NOT_INITIALIZED) return 0;
        return (state == StreamState.SHORTENED) ? terminationTimestamp : startTimestamp + streamDuration;
    }

    /// @return Current state of the stream.
    function getStreamState() external view returns (StreamState) {
        uint256 streamEnd = getStreamEnd();
        if (streamEnd == 0) return StreamState.NOT_INITIALIZED;
        return block.timestamp < streamEnd ? state : StreamState.FINISHED;
    }

    /** @notice Scales an amount from one decimal representation to another.
     * @param amount The amount to be scaled.
     * @param fromDecimals The number of decimals of the original amount.
     * @param toDecimals The number of decimals of the target amount.
     * @return The scaled amount.
     */
    function scaleAmount(uint256 amount, uint256 fromDecimals, uint256 toDecimals) internal pure returns (uint256) {
        if (fromDecimals == toDecimals) return amount;
        if (fromDecimals > toDecimals) {
            return amount / (10 ** (fromDecimals - toDecimals));
        }
        return amount * (10 ** (toDecimals - fromDecimals));
    }
}


// ===== FILE: contracts/StreamerFactory.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

import { AggregatorV3Interface } from "./interfaces/AggregatorV3Interface.sol";
import { IERC20 } from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import { IERC20Metadata } from "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import { Create2 } from "@openzeppelin/contracts/utils/Create2.sol";
import { IStreamerFactory } from "./interfaces/IStreamerFactory.sol";
import { Streamer } from "./Streamer.sol";

/** @title Streamer Factory
 * @author WOOF! Software
 * @custom:security-contact dmitriy@woof.software
 * @notice A Factory smart contract used for a safe deployment of new Streamer instances.
 * Anyone can use this Smart contract to deploy new streamers.
 */
contract StreamerFactory is IStreamerFactory {
    /// @notice A number per deployer used to generate a unique salt for Create2.
    mapping(address => uint256) public counters;

    /// @notice Deploys a new Streamer instance.
    /// @dev For details of each parameter, check documentation for Streamer.
    /// @dev Do not send tokens to Streamer address precomputed before actual deployment. Use the address returned from the function.
    /// @return The address of a new Streamer instance.
    function deployStreamer(
        address _streamingAsset,
        address _nativeAsset,
        AggregatorV3Interface _streamingAssetOracle,
        AggregatorV3Interface _nativeAssetOracle,
        address _returnAddress,
        address _streamCreator,
        address _recipient,
        uint256 _nativeAssetStreamingAmount,
        uint256 _slippage,
        uint256 _claimCooldown,
        uint256 _sweepCooldown,
        uint256 _streamDuration,
        uint256 _minimumNoticePeriod
    ) external returns (address) {
        if (_streamingAsset == _nativeAsset) revert AssetsMatch();
        uint8 streamingAssetDecimals = IERC20Metadata(_streamingAsset).decimals();
        uint8 nativeAssetDecimals = IERC20Metadata(_nativeAsset).decimals();
        bytes memory constructorParams = abi.encode(
            IERC20(_streamingAsset),
            _streamingAssetOracle,
            _nativeAssetOracle,
            _returnAddress,
            _streamCreator,
            _recipient,
            streamingAssetDecimals,
            nativeAssetDecimals,
            _nativeAssetStreamingAmount,
            _slippage,
            _claimCooldown,
            _sweepCooldown,
            _streamDuration,
            _minimumNoticePeriod
        );
        bytes32 uniqueSalt = keccak256(abi.encode(msg.sender, counters[msg.sender]++));
        bytes memory bytecodeWithParams = abi.encodePacked(type(Streamer).creationCode, constructorParams);
        address newContract = Create2.deploy(0, uniqueSalt, bytecodeWithParams);

        emit StreamerDeployed(newContract, constructorParams);
        return newContract;
    }
}
