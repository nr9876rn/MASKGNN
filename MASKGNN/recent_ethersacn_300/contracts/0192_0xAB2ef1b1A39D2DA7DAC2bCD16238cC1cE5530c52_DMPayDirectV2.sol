// ===== FILE: npm/_openzeppelin/contracts_5.6.1/access/Ownable.sol =====
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


// ===== FILE: npm/_openzeppelin/contracts_5.6.1/interfaces/draft-IERC6093.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.5.0) (interfaces/draft-IERC6093.sol)

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
     * @dev Indicates that an address can't be an owner. For example, `address(0)` is a forbidden owner in ERC-721.
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


// ===== FILE: npm/_openzeppelin/contracts_5.6.1/interfaces/IERC1363.sol =====
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


// ===== FILE: npm/_openzeppelin/contracts_5.6.1/interfaces/IERC165.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (interfaces/IERC165.sol)

pragma solidity >=0.4.16;

import {IERC165} from "../utils/introspection/IERC165.sol";


// ===== FILE: npm/_openzeppelin/contracts_5.6.1/interfaces/IERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (interfaces/IERC20.sol)

pragma solidity >=0.4.16;

import {IERC20} from "../token/ERC20/IERC20.sol";


// ===== FILE: npm/_openzeppelin/contracts_5.6.1/token/ERC20/ERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.5.0) (token/ERC20/ERC20.sol)

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
     * `_spendAllowance` during the `transferFrom` operation sets the flag to false. This saves gas by not emitting any
     * `Approval` event during `transferFrom` operations.
     *
     * Anyone who wishes to continue emitting `Approval` events on the `transferFrom` operation can force the flag to
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


// ===== FILE: npm/_openzeppelin/contracts_5.6.1/token/ERC20/extensions/IERC20Metadata.sol =====
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


// ===== FILE: npm/_openzeppelin/contracts_5.6.1/token/ERC20/IERC20.sol =====
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


// ===== FILE: npm/_openzeppelin/contracts_5.6.1/token/ERC20/utils/SafeERC20.sol =====
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


// ===== FILE: npm/_openzeppelin/contracts_5.6.1/utils/Context.sol =====
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


// ===== FILE: npm/_openzeppelin/contracts_5.6.1/utils/introspection/IERC165.sol =====
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


// ===== FILE: npm/_openzeppelin/contracts_5.6.1/utils/ReentrancyGuard.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.5.0) (utils/ReentrancyGuard.sol)

pragma solidity ^0.8.20;

import {StorageSlot} from "./StorageSlot.sol";

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
 *
 * IMPORTANT: Deprecated. This storage-based reentrancy guard will be removed and replaced
 * by the {ReentrancyGuardTransient} variant in v6.0.
 *
 * @custom:stateless
 */
abstract contract ReentrancyGuard {
    using StorageSlot for bytes32;

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.ReentrancyGuard")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant REENTRANCY_GUARD_STORAGE =
        0x9b779b17422d0df92223018b32b4d1fa46e071723d6817e2486d003becc55f00;

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

    /**
     * @dev Unauthorized reentrant call.
     */
    error ReentrancyGuardReentrantCall();

    constructor() {
        _reentrancyGuardStorageSlot().getUint256Slot().value = NOT_ENTERED;
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

    /**
     * @dev A `view` only version of {nonReentrant}. Use to block view functions
     * from being called, preventing reading from inconsistent contract state.
     *
     * CAUTION: This is a "view" modifier and does not change the reentrancy
     * status. Use it only on view functions. For payable or non-payable functions,
     * use the standard {nonReentrant} modifier instead.
     */
    modifier nonReentrantView() {
        _nonReentrantBeforeView();
        _;
    }

    function _nonReentrantBeforeView() private view {
        if (_reentrancyGuardEntered()) {
            revert ReentrancyGuardReentrantCall();
        }
    }

    function _nonReentrantBefore() private {
        // On the first call to nonReentrant, _status will be NOT_ENTERED
        _nonReentrantBeforeView();

        // Any calls to nonReentrant after this point will fail
        _reentrancyGuardStorageSlot().getUint256Slot().value = ENTERED;
    }

    function _nonReentrantAfter() private {
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _reentrancyGuardStorageSlot().getUint256Slot().value = NOT_ENTERED;
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        return _reentrancyGuardStorageSlot().getUint256Slot().value == ENTERED;
    }

    function _reentrancyGuardStorageSlot() internal pure virtual returns (bytes32) {
        return REENTRANCY_GUARD_STORAGE;
    }
}


// ===== FILE: npm/_openzeppelin/contracts_5.6.1/utils/StorageSlot.sol =====
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


// ===== FILE: project/contracts/DMPayDirect.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/// @title DMPayDirect
/// @notice Pay-to-DM in USDC or ETH. No registry, no handle — recipients are addresses.
///         Supports per-conversation pricing, lifetime "always message" passes,
///         and one-time-payment group chats. Identity resolved off-chain via ENS.
contract DMPayDirect is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    IERC20 public immutable usdc;
    address public treasury;

    uint256 public constant FEE_BPS = 250;       // 2.5%
    uint256 public constant BPS_BASE = 10000;

    // --- 1:1 pricing ----------------------------------------------------------

    struct Price {
        uint256 usdc;          // per-conversation USDC (0 = disabled)
        uint256 eth;           // per-conversation ETH  (0 = disabled)
        uint256 lifetimeUsdc;  // pay once for forever access (0 = disabled)
        uint256 lifetimeEth;   // pay once for forever access (0 = disabled)
    }
    mapping(address => Price) public priceOf;
    mapping(address => mapping(address => bool)) public hasLifetimePass; // recipient => sender => pass

    // --- Groups ---------------------------------------------------------------

    struct Group {
        address creator;
        uint256 priceUsdc;
        uint256 priceEth;
        uint64 capacity;       // 0 = unlimited
        uint64 memberCount;
        bool active;
        bytes32 xmtpGroupId;   // optional linkage to XMTP group, set by creator
    }
    mapping(uint256 => Group) public groups;
    mapping(uint256 => mapping(address => bool)) public isGroupMember;
    uint256 public nextGroupId;

    // --- Accounting -----------------------------------------------------------

    uint256 public accumulatedEthFees;

    // --- Events ---------------------------------------------------------------

    event PriceSet(address indexed user, uint256 usdc, uint256 eth, uint256 lifetimeUsdc, uint256 lifetimeEth);
    event ConversationOpened(address indexed sender, address indexed recipient, address indexed token, uint256 amountPaid, uint256 fee);
    event MessagePaid(address indexed sender, address indexed recipient, address indexed token, uint256 amountPaid, uint256 fee);
    event LifetimePassPurchased(address indexed sender, address indexed recipient, address indexed token, uint256 amountPaid, uint256 fee);

    event GroupCreated(uint256 indexed id, address indexed creator, uint256 priceUsdc, uint256 priceEth, uint64 capacity);
    event GroupXmtpIdSet(uint256 indexed id, bytes32 xmtpGroupId);
    event GroupJoined(uint256 indexed id, address indexed member, address indexed token, uint256 amountPaid, uint256 fee);
    event GroupClosed(uint256 indexed id);

    event TreasuryUpdated(address indexed newTreasury);

    constructor(address _usdc, address _treasury) Ownable(msg.sender) {
        require(_usdc != address(0), "usdc=0");
        require(_treasury != address(0), "treasury=0");
        usdc = IERC20(_usdc);
        treasury = _treasury;
    }

    // ============================================================
    // Recipient config
    // ============================================================

    /// @notice Set all four price tiers. Pass 0 to disable a tier.
    function setPrice(uint256 _usdc, uint256 _eth, uint256 _lifetimeUsdc, uint256 _lifetimeEth) external {
        priceOf[msg.sender] = Price(_usdc, _eth, _lifetimeUsdc, _lifetimeEth);
        emit PriceSet(msg.sender, _usdc, _eth, _lifetimeUsdc, _lifetimeEth);
    }

    // ============================================================
    // 1:1 payments
    // ============================================================

    function openConversationUSDC(address recipient) external nonReentrant {
        uint256 price = priceOf[recipient].usdc;
        require(price > 0, "USDC not accepted");
        _payUSDC(recipient, price);
        emit ConversationOpened(msg.sender, recipient, address(usdc), price, _feeOf(price));
    }

    function openConversationETH(address recipient) external payable nonReentrant {
        uint256 price = priceOf[recipient].eth;
        require(price > 0, "ETH not accepted");
        require(msg.value == price, "wrong eth amount");
        _payETH(recipient, price);
        emit ConversationOpened(msg.sender, recipient, address(0), price, _feeOf(price));
    }

    function payMessageUSDC(address recipient, uint256 amount) external nonReentrant {
        require(amount > 0, "amount=0");
        _payUSDC(recipient, amount);
        emit MessagePaid(msg.sender, recipient, address(usdc), amount, _feeOf(amount));
    }

    function payMessageETH(address recipient) external payable nonReentrant {
        require(msg.value > 0, "amount=0");
        _payETH(recipient, msg.value);
        emit MessagePaid(msg.sender, recipient, address(0), msg.value, _feeOf(msg.value));
    }

    // ============================================================
    // Lifetime passes
    // ============================================================

    function buyLifetimePassUSDC(address recipient) external nonReentrant {
        uint256 price = priceOf[recipient].lifetimeUsdc;
        require(price > 0, "Lifetime USDC not offered");
        require(!hasLifetimePass[recipient][msg.sender], "already has pass");
        hasLifetimePass[recipient][msg.sender] = true;
        _payUSDC(recipient, price);
        emit LifetimePassPurchased(msg.sender, recipient, address(usdc), price, _feeOf(price));
    }

    function buyLifetimePassETH(address recipient) external payable nonReentrant {
        uint256 price = priceOf[recipient].lifetimeEth;
        require(price > 0, "Lifetime ETH not offered");
        require(msg.value == price, "wrong eth amount");
        require(!hasLifetimePass[recipient][msg.sender], "already has pass");
        hasLifetimePass[recipient][msg.sender] = true;
        _payETH(recipient, price);
        emit LifetimePassPurchased(msg.sender, recipient, address(0), price, _feeOf(price));
    }

    // ============================================================
    // Groups
    // ============================================================

    function createGroup(uint256 priceUsdc, uint256 priceEth, uint64 capacity) external returns (uint256 id) {
        require(priceUsdc > 0 || priceEth > 0, "no price set");
        id = nextGroupId++;
        groups[id] = Group({
            creator: msg.sender,
            priceUsdc: priceUsdc,
            priceEth: priceEth,
            capacity: capacity,
            memberCount: 1,
            active: true,
            xmtpGroupId: bytes32(0)
        });
        isGroupMember[id][msg.sender] = true;
        emit GroupCreated(id, msg.sender, priceUsdc, priceEth, capacity);
    }

    function setGroupXmtpId(uint256 id, bytes32 xmtpGroupId) external {
        Group storage g = groups[id];
        require(g.creator == msg.sender, "not creator");
        g.xmtpGroupId = xmtpGroupId;
        emit GroupXmtpIdSet(id, xmtpGroupId);
    }

    function closeGroup(uint256 id) external {
        Group storage g = groups[id];
        require(g.creator == msg.sender, "not creator");
        g.active = false;
        emit GroupClosed(id);
    }

    function joinGroupUSDC(uint256 id) external nonReentrant {
        Group storage g = groups[id];
        _preJoin(g, id);
        uint256 price = g.priceUsdc;
        require(price > 0, "USDC not accepted");
        _payUSDC(g.creator, price);
        isGroupMember[id][msg.sender] = true;
        unchecked { g.memberCount += 1; }
        emit GroupJoined(id, msg.sender, address(usdc), price, _feeOf(price));
    }

    function joinGroupETH(uint256 id) external payable nonReentrant {
        Group storage g = groups[id];
        _preJoin(g, id);
        uint256 price = g.priceEth;
        require(price > 0, "ETH not accepted");
        require(msg.value == price, "wrong eth amount");
        _payETH(g.creator, price);
        isGroupMember[id][msg.sender] = true;
        unchecked { g.memberCount += 1; }
        emit GroupJoined(id, msg.sender, address(0), price, _feeOf(price));
    }

    function _preJoin(Group storage g, uint256 id) internal view {
        require(g.creator != address(0), "no group");
        require(g.active, "group closed");
        require(!isGroupMember[id][msg.sender], "already member");
        require(g.capacity == 0 || g.memberCount < g.capacity, "group full");
    }

    // ============================================================
    // Internal payment helpers
    // ============================================================

    function _feeOf(uint256 amount) internal pure returns (uint256) {
        return (amount * FEE_BPS) / BPS_BASE;
    }

    function _payUSDC(address recipient, uint256 amount) internal {
        uint256 fee = _feeOf(amount);
        uint256 net = amount - fee;
        usdc.safeTransferFrom(msg.sender, recipient, net);
        if (fee > 0) usdc.safeTransferFrom(msg.sender, treasury, fee);
    }

    function _payETH(address recipient, uint256 amount) internal {
        uint256 fee = _feeOf(amount);
        uint256 net = amount - fee;
        (bool ok, ) = recipient.call{value: net}("");
        require(ok, "eth to recipient failed");
        if (fee > 0) accumulatedEthFees += fee;
    }

    // ============================================================
    // Admin
    // ============================================================

    function setTreasury(address _treasury) external onlyOwner {
        require(_treasury != address(0), "treasury=0");
        treasury = _treasury;
        emit TreasuryUpdated(_treasury);
    }

    function withdrawEthFees() external onlyOwner nonReentrant {
        uint256 amount = accumulatedEthFees;
        accumulatedEthFees = 0;
        (bool ok, ) = treasury.call{value: amount}("");
        require(ok, "eth withdraw failed");
    }
}


// ===== FILE: project/contracts/DMPayDirectV2.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/// @title DMPayDirectV2
/// @notice V2 adds receiver-side block / close primitives and lifetime-pass
///         bypass. Lifetime pass holders cannot be blocked or closed, and they
///         join the creator's paid groups for free.
///
///         Migration from V1: state does not carry over. Recipients call
///         setPrice() once on V2 to re-enable paid DMs. V1 contract remains
///         callable; the dapp points at V2.
contract DMPayDirectV2 is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    IERC20 public immutable usdc;
    address public treasury;

    uint256 public constant FEE_BPS = 250;       // 2.5%
    uint256 public constant BPS_BASE = 10000;

    // --- 1:1 pricing ----------------------------------------------------------

    struct Price {
        uint256 usdc;
        uint256 eth;
        uint256 lifetimeUsdc;
        uint256 lifetimeEth;
    }
    mapping(address => Price) public priceOf;
    mapping(address => mapping(address => bool)) public hasLifetimePass; // recipient => sender => pass

    // --- V2: receiver-side controls -------------------------------------------

    /// @dev recipient => sender => permanent block (lifetime bypasses).
    mapping(address => mapping(address => bool)) public blockedSenders;

    /// @dev recipient => sender => timestamp the receiver last closed.
    ///      Sender's open is considered fresh iff openedAt > closedAt.
    mapping(address => mapping(address => uint64)) public closedAt;

    /// @dev recipient => sender => timestamp of sender's last open.
    mapping(address => mapping(address => uint64)) public openedAt;

    // --- Groups ---------------------------------------------------------------

    struct Group {
        address creator;
        uint256 priceUsdc;
        uint256 priceEth;
        uint64 capacity;       // 0 = unlimited
        uint64 memberCount;
        bool active;
        bytes32 xmtpGroupId;
    }
    mapping(uint256 => Group) public groups;
    mapping(uint256 => mapping(address => bool)) public isGroupMember;
    uint256 public nextGroupId;

    // --- Accounting -----------------------------------------------------------

    uint256 public accumulatedEthFees;

    // --- Events ---------------------------------------------------------------

    event PriceSet(address indexed user, uint256 usdc, uint256 eth, uint256 lifetimeUsdc, uint256 lifetimeEth);
    event ConversationOpened(address indexed sender, address indexed recipient, address indexed token, uint256 amountPaid, uint256 fee);
    event MessagePaid(address indexed sender, address indexed recipient, address indexed token, uint256 amountPaid, uint256 fee);
    event LifetimePassPurchased(address indexed sender, address indexed recipient, address indexed token, uint256 amountPaid, uint256 fee);

    event SenderBlocked(address indexed recipient, address indexed sender);
    event SenderUnblocked(address indexed recipient, address indexed sender);
    event ConversationClosed(address indexed recipient, address indexed sender, uint64 closedAt);

    event GroupCreated(uint256 indexed id, address indexed creator, uint256 priceUsdc, uint256 priceEth, uint64 capacity);
    event GroupXmtpIdSet(uint256 indexed id, bytes32 xmtpGroupId);
    event GroupJoined(uint256 indexed id, address indexed member, address indexed token, uint256 amountPaid, uint256 fee);
    event GroupMemberRemoved(uint256 indexed id, address indexed member);
    event GroupClosed(uint256 indexed id);

    event TreasuryUpdated(address indexed newTreasury);

    constructor(address _usdc, address _treasury) Ownable(msg.sender) {
        require(_usdc != address(0), "usdc=0");
        require(_treasury != address(0), "treasury=0");
        usdc = IERC20(_usdc);
        treasury = _treasury;
    }

    // ============================================================
    // Recipient config
    // ============================================================

    function setPrice(uint256 _usdc, uint256 _eth, uint256 _lifetimeUsdc, uint256 _lifetimeEth) external {
        priceOf[msg.sender] = Price(_usdc, _eth, _lifetimeUsdc, _lifetimeEth);
        emit PriceSet(msg.sender, _usdc, _eth, _lifetimeUsdc, _lifetimeEth);
    }

    // ============================================================
    // Receiver-side block / close
    // ============================================================

    /// @notice Permanently block `sender` from opening / paying / joining your
    ///         groups. Lifetime pass holders are exempt.
    function blockSender(address sender) external {
        blockedSenders[msg.sender][sender] = true;
        emit SenderBlocked(msg.sender, sender);
    }

    function unblockSender(address sender) external {
        blockedSenders[msg.sender][sender] = false;
        emit SenderUnblocked(msg.sender, sender);
    }

    /// @notice Close `sender`'s current open. They'll need to pay again to
    ///         re-open. Lifetime pass holders are exempt (their unlock survives).
    function closeConversation(address sender) external {
        uint64 ts = uint64(block.timestamp);
        closedAt[msg.sender][sender] = ts;
        emit ConversationClosed(msg.sender, sender, ts);
    }

    /// @notice True if `sender` has an active 1:1 unlock with `recipient`.
    function isUnlocked(address recipient, address sender) external view returns (bool) {
        return _isUnlocked(recipient, sender);
    }

    function _isUnlocked(address recipient, address sender) internal view returns (bool) {
        if (hasLifetimePass[recipient][sender]) return true;
        if (blockedSenders[recipient][sender]) return false;
        uint64 opened = openedAt[recipient][sender];
        if (opened == 0) return false;
        return opened > closedAt[recipient][sender];
    }

    // ============================================================
    // 1:1 payments
    // ============================================================

    function openConversationUSDC(address recipient) external nonReentrant {
        _requireNotBlocked(recipient);
        uint256 price = priceOf[recipient].usdc;
        require(price > 0, "USDC not accepted");
        _payUSDC(recipient, price);
        openedAt[recipient][msg.sender] = uint64(block.timestamp);
        emit ConversationOpened(msg.sender, recipient, address(usdc), price, _feeOf(price));
    }

    function openConversationETH(address recipient) external payable nonReentrant {
        _requireNotBlocked(recipient);
        uint256 price = priceOf[recipient].eth;
        require(price > 0, "ETH not accepted");
        require(msg.value == price, "wrong eth amount");
        _payETH(recipient, price);
        openedAt[recipient][msg.sender] = uint64(block.timestamp);
        emit ConversationOpened(msg.sender, recipient, address(0), price, _feeOf(price));
    }

    function payMessageUSDC(address recipient, uint256 amount) external nonReentrant {
        _requireNotBlocked(recipient);
        require(amount > 0, "amount=0");
        _payUSDC(recipient, amount);
        emit MessagePaid(msg.sender, recipient, address(usdc), amount, _feeOf(amount));
    }

    function payMessageETH(address recipient) external payable nonReentrant {
        _requireNotBlocked(recipient);
        require(msg.value > 0, "amount=0");
        _payETH(recipient, msg.value);
        emit MessagePaid(msg.sender, recipient, address(0), msg.value, _feeOf(msg.value));
    }

    // ============================================================
    // Lifetime passes
    // ============================================================
    // NOTE: intentionally NOT gated on blockedSenders. Buying a lifetime pass
    // is the escape hatch from a block — if a recipient never wants to be
    // overrideable, they can disable both lifetime tiers via setPrice(.., 0, 0).

    function buyLifetimePassUSDC(address recipient) external nonReentrant {
        uint256 price = priceOf[recipient].lifetimeUsdc;
        require(price > 0, "Lifetime USDC not offered");
        require(!hasLifetimePass[recipient][msg.sender], "already has pass");
        hasLifetimePass[recipient][msg.sender] = true;
        _payUSDC(recipient, price);
        emit LifetimePassPurchased(msg.sender, recipient, address(usdc), price, _feeOf(price));
    }

    function buyLifetimePassETH(address recipient) external payable nonReentrant {
        uint256 price = priceOf[recipient].lifetimeEth;
        require(price > 0, "Lifetime ETH not offered");
        require(msg.value == price, "wrong eth amount");
        require(!hasLifetimePass[recipient][msg.sender], "already has pass");
        hasLifetimePass[recipient][msg.sender] = true;
        _payETH(recipient, price);
        emit LifetimePassPurchased(msg.sender, recipient, address(0), price, _feeOf(price));
    }

    // ============================================================
    // Groups
    // ============================================================

    function createGroup(uint256 priceUsdc, uint256 priceEth, uint64 capacity) external returns (uint256 id) {
        require(priceUsdc > 0 || priceEth > 0, "no price set");
        id = nextGroupId++;
        groups[id] = Group({
            creator: msg.sender,
            priceUsdc: priceUsdc,
            priceEth: priceEth,
            capacity: capacity,
            memberCount: 1,
            active: true,
            xmtpGroupId: bytes32(0)
        });
        isGroupMember[id][msg.sender] = true;
        emit GroupCreated(id, msg.sender, priceUsdc, priceEth, capacity);
    }

    function setGroupXmtpId(uint256 id, bytes32 xmtpGroupId) external {
        Group storage g = groups[id];
        require(g.creator == msg.sender, "not creator");
        g.xmtpGroupId = xmtpGroupId;
        emit GroupXmtpIdSet(id, xmtpGroupId);
    }

    function closeGroup(uint256 id) external {
        Group storage g = groups[id];
        require(g.creator == msg.sender, "not creator");
        g.active = false;
        emit GroupClosed(id);
    }

    /// @notice Creator can evict a member. Removed member can re-join by paying
    ///         again (lifetime pass holders re-join free).
    function removeGroupMember(uint256 id, address member) external {
        Group storage g = groups[id];
        require(g.creator == msg.sender, "not creator");
        require(member != g.creator, "cannot remove creator");
        require(isGroupMember[id][member], "not member");
        isGroupMember[id][member] = false;
        unchecked { g.memberCount -= 1; }
        emit GroupMemberRemoved(id, member);
    }

    function joinGroupUSDC(uint256 id) external nonReentrant {
        Group storage g = groups[id];
        _preJoin(g, id);
        if (hasLifetimePass[g.creator][msg.sender]) {
            // Lifetime pass holder: free entry, no token transfer.
            isGroupMember[id][msg.sender] = true;
            unchecked { g.memberCount += 1; }
            emit GroupJoined(id, msg.sender, address(0), 0, 0);
            return;
        }
        require(!blockedSenders[g.creator][msg.sender], "blocked");
        uint256 price = g.priceUsdc;
        require(price > 0, "USDC not accepted");
        _payUSDC(g.creator, price);
        isGroupMember[id][msg.sender] = true;
        unchecked { g.memberCount += 1; }
        emit GroupJoined(id, msg.sender, address(usdc), price, _feeOf(price));
    }

    function joinGroupETH(uint256 id) external payable nonReentrant {
        Group storage g = groups[id];
        _preJoin(g, id);
        if (hasLifetimePass[g.creator][msg.sender]) {
            require(msg.value == 0, "lifetime: send no eth");
            isGroupMember[id][msg.sender] = true;
            unchecked { g.memberCount += 1; }
            emit GroupJoined(id, msg.sender, address(0), 0, 0);
            return;
        }
        require(!blockedSenders[g.creator][msg.sender], "blocked");
        uint256 price = g.priceEth;
        require(price > 0, "ETH not accepted");
        require(msg.value == price, "wrong eth amount");
        _payETH(g.creator, price);
        isGroupMember[id][msg.sender] = true;
        unchecked { g.memberCount += 1; }
        emit GroupJoined(id, msg.sender, address(0), price, _feeOf(price));
    }

    function _preJoin(Group storage g, uint256 id) internal view {
        require(g.creator != address(0), "no group");
        require(g.active, "group closed");
        require(!isGroupMember[id][msg.sender], "already member");
        require(g.capacity == 0 || g.memberCount < g.capacity, "group full");
    }

    // ============================================================
    // Internal payment helpers
    // ============================================================

    function _requireNotBlocked(address recipient) internal view {
        if (blockedSenders[recipient][msg.sender] && !hasLifetimePass[recipient][msg.sender]) {
            revert("blocked");
        }
    }

    function _feeOf(uint256 amount) internal pure returns (uint256) {
        return (amount * FEE_BPS) / BPS_BASE;
    }

    function _payUSDC(address recipient, uint256 amount) internal {
        uint256 fee = _feeOf(amount);
        uint256 net = amount - fee;
        usdc.safeTransferFrom(msg.sender, recipient, net);
        if (fee > 0) usdc.safeTransferFrom(msg.sender, treasury, fee);
    }

    function _payETH(address recipient, uint256 amount) internal {
        uint256 fee = _feeOf(amount);
        uint256 net = amount - fee;
        (bool ok, ) = recipient.call{value: net}("");
        require(ok, "eth to recipient failed");
        if (fee > 0) accumulatedEthFees += fee;
    }

    // ============================================================
    // Admin
    // ============================================================

    function setTreasury(address _treasury) external onlyOwner {
        require(_treasury != address(0), "treasury=0");
        treasury = _treasury;
        emit TreasuryUpdated(_treasury);
    }

    function withdrawEthFees() external onlyOwner nonReentrant {
        uint256 amount = accumulatedEthFees;
        accumulatedEthFees = 0;
        (bool ok, ) = treasury.call{value: amount}("");
        require(ok, "eth withdraw failed");
    }
}


// ===== FILE: project/contracts/DMPayMessaging.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

interface IDMPayRegistry {
    struct UserProfile {
        address wallet;
        string xHandle;
        string bio;
        string pfpUrl;
        uint256 priceUSDC;
        string ipfsHash;
        bool registered;
        bool active;
    }
    function getProfileByWallet(address wallet) external view returns (UserProfile memory);
    function getProfile(string calldata xHandle) external view returns (UserProfile memory);
}

contract DMPayMessaging is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    IERC20 public usdc;
    IDMPayRegistry public registry;

    // 2.5% fee = 250 basis points
    uint256 public constant FEE_BPS = 250;
    uint256 public constant BPS_BASE = 10000;

    enum ConversationStatus { Closed, Open }

    struct Conversation {
        address sender;
        address recipient;
        uint256 totalPaid;
        uint256 lastPayment;
        ConversationStatus status;
        uint256 openedAt;
        uint256 closedAt;
        uint256 messageCount;
    }

    // conversationId => Conversation
    mapping(bytes32 => Conversation) public conversations;
    // sender => recipient => conversationId
    mapping(address => mapping(address => bytes32)) public activeConversation;

    // accumulated fees for owner to withdraw
    uint256 public accumulatedFees;

    event ConversationOpened(
        bytes32 indexed conversationId,
        address indexed sender,
        address indexed recipient,
        uint256 amountPaid,
        uint256 fee
    );

    event MessagePaid(
        bytes32 indexed conversationId,
        address indexed sender,
        address indexed recipient,
        uint256 amountPaid,
        uint256 fee
    );

    event ConversationClosed(
        bytes32 indexed conversationId,
        address indexed closedBy,
        address indexed recipient
    );

    constructor(address _usdc, address _registry) Ownable(msg.sender) {
        usdc = IERC20(_usdc);
        registry = IDMPayRegistry(_registry);
    }

    function getConversationId(address sender, address recipient) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(sender, recipient));
    }

    function calculateFee(uint256 amount) public pure returns (uint256 fee, uint256 net) {
        fee = (amount * FEE_BPS) / BPS_BASE;
        net = amount - fee;
    }

    function openConversation(address recipient) external nonReentrant {
        require(recipient != msg.sender, "Cannot message yourself");

        // Get recipient price from registry
        IDMPayRegistry.UserProfile memory profile = registry.getProfileByWallet(recipient);
        require(profile.registered, "Recipient not registered");
        require(profile.active, "Recipient not active");
        require(profile.priceUSDC > 0, "Recipient has no price set");

        bytes32 convId = getConversationId(msg.sender, recipient);
        Conversation storage conv = conversations[convId];

        require(conv.status == ConversationStatus.Closed, "Conversation already open");

        uint256 price = profile.priceUSDC;
        (uint256 fee, uint256 net) = calculateFee(price);

        // Transfer USDC from sender
        usdc.safeTransferFrom(msg.sender, address(this), price);

        // Pay recipient net amount
        usdc.safeTransfer(recipient, net);

        // Accumulate fee
        accumulatedFees += fee;

        // Update conversation
        conv.sender = msg.sender;
        conv.recipient = recipient;
        conv.totalPaid += price;
        conv.lastPayment = block.timestamp;
        conv.status = ConversationStatus.Open;
        conv.openedAt = block.timestamp;
        conv.messageCount += 1;

        activeConversation[msg.sender][recipient] = convId;

        emit ConversationOpened(convId, msg.sender, recipient, price, fee);
    }

    function payForMessage(address recipient) external nonReentrant {
        bytes32 convId = getConversationId(msg.sender, recipient);
        Conversation storage conv = conversations[convId];

        require(conv.status == ConversationStatus.Open, "Conversation not open");
        require(conv.sender == msg.sender, "Not conversation sender");

        IDMPayRegistry.UserProfile memory profile = registry.getProfileByWallet(recipient);
        require(profile.registered, "Recipient not registered");

        uint256 price = profile.priceUSDC;
        (uint256 fee, uint256 net) = calculateFee(price);

        usdc.safeTransferFrom(msg.sender, address(this), price);
        usdc.safeTransfer(recipient, net);
        accumulatedFees += fee;

        conv.totalPaid += price;
        conv.lastPayment = block.timestamp;
        conv.messageCount += 1;

        emit MessagePaid(convId, msg.sender, recipient, price, fee);
    }

    function closeConversation(address sender) external nonReentrant {
        bytes32 convId = getConversationId(sender, msg.sender);
        Conversation storage conv = conversations[convId];

        require(conv.status == ConversationStatus.Open, "Conversation not open");
        require(conv.recipient == msg.sender, "Only recipient can close");

        conv.status = ConversationStatus.Closed;
        conv.closedAt = block.timestamp;

        emit ConversationClosed(convId, msg.sender, msg.sender);
    }

    function getConversation(address sender, address recipient)
        external
        view
        returns (Conversation memory)
    {
        bytes32 convId = getConversationId(sender, recipient);
        return conversations[convId];
    }

    function isConversationOpen(address sender, address recipient)
        external
        view
        returns (bool)
    {
        bytes32 convId = getConversationId(sender, recipient);
        return conversations[convId].status == ConversationStatus.Open;
    }

    // Owner withdraws accumulated fees
    function withdrawFees() external onlyOwner nonReentrant {
        uint256 amount = accumulatedFees;
        require(amount > 0, "No fees to withdraw");
        accumulatedFees = 0;
        usdc.safeTransfer(owner(), amount);
    }

    function setRegistry(address _registry) external onlyOwner {
        registry = IDMPayRegistry(_registry);
    }

    function setUSDC(address _usdc) external onlyOwner {
        usdc = IERC20(_usdc);
    }
}


// ===== FILE: project/contracts/DMPayRegistry.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

interface IENSRegistry {
    function setSubnodeOwner(bytes32 node, bytes32 label, address owner) external returns (bytes32);
    function setSubnodeRecord(bytes32 node, bytes32 label, address owner, address resolver, uint64 ttl) external;
    function owner(bytes32 node) external view returns (address);
    function isApprovedForAll(address owner, address operator) external view returns (bool);
    function setOwner(bytes32 node, address owner) external;
}

interface IENSResolver {
    function setContenthash(bytes32 node, bytes memory hash) external;
    function setAddr(bytes32 node, address addr) external;
}

contract DMPayRegistry is Ownable, ReentrancyGuard {

    bytes32 public parentNode;
    IENSRegistry public ensRegistry;
    IENSResolver public ensResolver;

    struct UserProfile {
        address wallet;
        string xHandle;
        string bio;
        string pfpUrl;
        uint256 priceUSDC;
        string ipfsHash;
        bool registered;
        bool active;
    }

    mapping(string => UserProfile) public profiles;
    mapping(address => string) public walletToHandle;

    event ProfileRegistered(address indexed wallet, string xHandle, uint256 priceUSDC);
    event ProfileUpdated(address indexed wallet, string xHandle, string bio, string pfpUrl, uint256 priceUSDC);
    event IPFSHashUpdated(address indexed wallet, string xHandle, string ipfsHash);
    event SubdomainRegistered(address indexed wallet, string xHandle, bytes32 ensNode);

    constructor(
        bytes32 _parentNode,
        address _ensRegistry,
        address _ensResolver
    ) Ownable(msg.sender) {
        parentNode = _parentNode;
        ensRegistry = IENSRegistry(_ensRegistry);
        ensResolver = IENSResolver(_ensResolver);
    }

    function toLower(string memory str) internal pure returns (string memory) {
        bytes memory bStr = bytes(str);
        bytes memory bLower = new bytes(bStr.length);
        for (uint i = 0; i < bStr.length; i++) {
            if ((uint8(bStr[i]) >= 65) && (uint8(bStr[i]) <= 90)) {
                bLower[i] = bytes1(uint8(bStr[i]) + 32);
            } else {
                bLower[i] = bStr[i];
            }
        }
        return string(bLower);
    }

    function getSubnode(string memory handle) public view returns (bytes32) {
        bytes32 label = keccak256(bytes(toLower(handle)));
        return keccak256(abi.encodePacked(parentNode, label));
    }

    function registerProfile(
        string calldata xHandle,
        string calldata bio,
        string calldata pfpUrl,
        uint256 priceUSDC
    ) external nonReentrant {
        require(bytes(xHandle).length > 0, "Handle required");
        require(priceUSDC > 0, "Price must be > 0");

        string memory handle = toLower(xHandle);
        require(!profiles[handle].registered, "Handle already registered");
        require(bytes(walletToHandle[msg.sender]).length == 0, "Wallet already registered");

        profiles[handle] = UserProfile({
            wallet: msg.sender,
            xHandle: xHandle,
            bio: bio,
            pfpUrl: pfpUrl,
            priceUSDC: priceUSDC,
            ipfsHash: "",
            registered: true,
            active: true
        });

        walletToHandle[msg.sender] = handle;
        emit ProfileRegistered(msg.sender, xHandle, priceUSDC);
        _tryRegisterSubdomain(handle, msg.sender);
    }

    function _tryRegisterSubdomain(string memory handle, address userWallet) internal {
        bytes32 label = keccak256(bytes(handle));
        bytes32 subnode = keccak256(abi.encodePacked(parentNode, label));

        try ensRegistry.setSubnodeRecord(
            parentNode,
            label,
            address(this),
            address(ensResolver),
            0
        ) {
            try ensResolver.setAddr(subnode, userWallet) {} catch {}
            try ensRegistry.setOwner(subnode, userWallet) {} catch {}
            emit SubdomainRegistered(userWallet, handle, subnode);
        } catch {
            try ensRegistry.setSubnodeOwner(parentNode, label, userWallet) returns (bytes32 node) {
                emit SubdomainRegistered(userWallet, handle, node);
            } catch {}
        }
    }

    // Frontend passes pre-encoded contenthash bytes
    function updateIPFSHash(string calldata ipfsHash, bytes calldata contenthash) external {
        string memory handle = walletToHandle[msg.sender];
        require(bytes(handle).length > 0, "Not registered");
        profiles[handle].ipfsHash = ipfsHash;

        bytes32 label = keccak256(bytes(handle));
        bytes32 subnode = keccak256(abi.encodePacked(parentNode, label));

        try ensRegistry.setSubnodeRecord(
            parentNode,
            label,
            address(this),
            address(ensResolver),
            0
        ) {
            if (contenthash.length > 0) {
                try ensResolver.setContenthash(subnode, contenthash) {} catch {}
            }
            try ensResolver.setAddr(subnode, msg.sender) {} catch {}
            try ensRegistry.setOwner(subnode, msg.sender) {} catch {}
        } catch {}

        emit IPFSHashUpdated(msg.sender, handle, ipfsHash);
    }

    function updateProfile(
        string calldata bio,
        string calldata pfpUrl,
        uint256 priceUSDC
    ) external {
        string memory handle = walletToHandle[msg.sender];
        require(bytes(handle).length > 0, "Not registered");
        require(priceUSDC > 0, "Price must be > 0");

        profiles[handle].bio = bio;
        profiles[handle].pfpUrl = pfpUrl;
        profiles[handle].priceUSDC = priceUSDC;

        emit ProfileUpdated(msg.sender, handle, bio, pfpUrl, priceUSDC);
    }

    function registerSubdomain() external {
        string memory handle = walletToHandle[msg.sender];
        require(bytes(handle).length > 0, "Not registered");

        bytes32 label = keccak256(bytes(handle));
        bytes32 subnode = keccak256(abi.encodePacked(parentNode, label));

        ensRegistry.setSubnodeRecord(parentNode, label, address(this), address(ensResolver), 0);
        ensResolver.setAddr(subnode, msg.sender);
        ensRegistry.setOwner(subnode, msg.sender);

        emit SubdomainRegistered(msg.sender, handle, subnode);
    }

    function getProfile(string calldata xHandle) external view returns (UserProfile memory) {
        return profiles[toLower(xHandle)];
    }

    function getProfileByWallet(address wallet) external view returns (UserProfile memory) {
        string memory handle = walletToHandle[wallet];
        return profiles[handle];
    }

    function setParentNode(bytes32 _parentNode) external onlyOwner {
        parentNode = _parentNode;
    }

    function setENSRegistry(address _ensRegistry) external onlyOwner {
        ensRegistry = IENSRegistry(_ensRegistry);
    }

    function setENSResolver(address _ensResolver) external onlyOwner {
        ensResolver = IENSResolver(_ensResolver);
    }
}


// ===== FILE: project/contracts/mocks/MockUSDC.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract MockUSDC is ERC20 {
    constructor() ERC20("Mock USDC", "USDC") {}
    function decimals() public pure override returns (uint8) { return 6; }
    function mint(address to, uint256 amount) external { _mint(to, amount); }
}
