// ===== FILE: _openzeppelin/contracts/access/Ownable.sol =====
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


// ===== FILE: _openzeppelin/contracts/access/Ownable2Step.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (access/Ownable2Step.sol)

pragma solidity ^0.8.20;

import {Ownable} from "./Ownable.sol";

/**
 * @dev Contract module which provides access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * This extension of the {Ownable} contract includes a two-step mechanism to transfer
 * ownership, where the new owner must call {acceptOwnership} in order to replace the
 * old one. This can help prevent common mistakes, such as transfers of ownership to
 * incorrect accounts, or to contracts that are unable to interact with the
 * permission system.
 *
 * The initial owner is specified at deployment time in the constructor for `Ownable`. This
 * can later be changed with {transferOwnership} and {acceptOwnership}.
 *
 * This module is used through inheritance. It will make available all functions
 * from parent (Ownable).
 */
abstract contract Ownable2Step is Ownable {
    address private _pendingOwner;

    event OwnershipTransferStarted(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev Returns the address of the pending owner.
     */
    function pendingOwner() public view virtual returns (address) {
        return _pendingOwner;
    }

    /**
     * @dev Starts the ownership transfer of the contract to a new account. Replaces the pending transfer if there is one.
     * Can only be called by the current owner.
     *
     * Setting `newOwner` to the zero address is allowed; this can be used to cancel an initiated ownership transfer.
     */
    function transferOwnership(address newOwner) public virtual override onlyOwner {
        _pendingOwner = newOwner;
        emit OwnershipTransferStarted(owner(), newOwner);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`) and deletes any pending owner.
     * Internal function without access restriction.
     */
    function _transferOwnership(address newOwner) internal virtual override {
        delete _pendingOwner;
        super._transferOwnership(newOwner);
    }

    /**
     * @dev The new owner accepts the ownership transfer.
     */
    function acceptOwnership() public virtual {
        address sender = _msgSender();
        if (pendingOwner() != sender) {
            revert OwnableUnauthorizedAccount(sender);
        }
        _transferOwnership(sender);
    }
}


// ===== FILE: _openzeppelin/contracts/interfaces/draft-IERC6093.sol =====
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


// ===== FILE: _openzeppelin/contracts/token/ERC20/ERC20.sol =====
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


// ===== FILE: _openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol =====
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


// ===== FILE: _openzeppelin/contracts/token/ERC20/IERC20.sol =====
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


// ===== FILE: contracts/fsc20/Factory.sol =====
// SPDX-License-Identifier: GPL-2.0-or-later
pragma solidity ^0.8.9;

import './FSC20.sol';
import '../walletList/interfaces/IWalletList.sol';

/**
 * @title FSC20Factory
 * @notice Factory contract for deploying FSC20 tokens with standardized WalletList integration
 * @dev Only the factory owner can deploy new tokens. Each deployed token shares the same WalletList contract
 *
 * Key Features:
 * - Centralized token deployment with consistent access control
 * - All deployed tokens use the same WalletList contract
 * - Ownership of deployed tokens is transferred to the deployer
 * - Event emission for tracking deployed tokens
 */
contract FSC20Factory is Ownable2Step {
  /// @notice The WalletList contract used by all deployed tokens
  IWalletList public immutable walletListContract;

  /**
   * @notice Emitted when a new FSC20 token is deployed
   * @param token The address of the newly deployed FSC20 token
   */
  event FSC20TokenCreated(address token);

  /**
   * @notice Thrown when the WalletList address provided to the constructor is zero or has no contract code
   * @param walletList The invalid address that was supplied
   */
  error InvalidWalletList(address walletList);

  /**
   * @notice Constructs a new FSC20Factory
   * @param walletListContract_ The WalletList contract address to be used by all deployed tokens
   */
  constructor(IWalletList walletListContract_) Ownable(_msgSender()) {
    address walletListAddr = address(walletListContract_);
    if (walletListAddr == address(0) || walletListAddr.code.length == 0) {
      revert InvalidWalletList(walletListAddr);
    }
    walletListContract = walletListContract_;
  }

  /**
   * @notice Deploys a new FSC20 token with the specified name and symbol
   * @dev Only callable by the factory owner. Ownership of the new token is transferred to msg.sender
   * @param name_ The name of the new token
   * @param symbol_ The symbol of the new token
   * @custom:emits FSC20TokenCreated with the address of the deployed token
   */
  function deploy(string memory name_, string memory symbol_) external onlyOwner {
    FSC20 token = new FSC20(name_, symbol_, walletListContract);
    token.transferOwnership(msg.sender);
    emit FSC20TokenCreated(address(token));
  }
}


// ===== FILE: contracts/fsc20/FSC20.sol =====
// SPDX-License-Identifier: GPL-2.0-or-later
pragma solidity ^0.8.9;

import '@openzeppelin/contracts/access/Ownable2Step.sol';
import '@openzeppelin/contracts/token/ERC20/ERC20.sol';
import '../walletList/interfaces/IWalletList.sol';
import './interfaces/IFSC20.sol';

/**
 * @title FSC20
 * @notice A compliant ERC20 token with whitelist, blacklist, freeze, pause, and document management capabilities
 * @dev Implements ERC20 with additional access control features using WalletList
 *
 * Key Features:
 * - Whitelist/Blacklist/Frozenlist support via WalletList contract
 * - Token freezing mechanism to lock tokens temporarily
 * - Pause functionality to halt all transfers and approvals
 * - Document management via IPFS CIDs
 * - Batch operations for gas efficiency
 *
 * Access Control Roles (managed by WalletList):
 * - MEMBER: Members who can manage whitelists and frozen lists
 * - WHITELIST: Addresses allowed to hold and transfer tokens
 * - FROZENLIST: Addresses whose transfers are frozen
 * - BLACKLIST: Addresses completely blocked from transactions
 */
contract FSC20 is IFSC20, ERC20, Ownable2Step {
  /// @notice The WalletList contract used for access control
  IWalletList public immutable walletListContract;

  /// @notice Whether the contract is currently paused
  bool public paused;

  /// @notice Mapping of addresses to their frozen token balances
  mapping(address => uint256) public frozenBalance;

  /// @notice Array of IPFS CIDs for associated documents
  string[] public ipfsCids;

  /// @notice Role identifier for members who can manage lists
  bytes32 public constant MEMBER = keccak256('MEMBER');

  /// @notice List identifier for whitelisted addresses
  bytes32 public constant WHITELIST = keccak256('WHITELIST');

  /// @notice List identifier for frozen addresses
  bytes32 public constant FROZENLIST = keccak256('FROZENLIST');

  /// @notice List identifier for blacklisted addresses
  bytes32 public constant BLACKLIST = keccak256('BLACKLIST');

  /**
   * @notice Constructs a new FSC20 token
   * @param name_ The name of the token
   * @param symbol_ The symbol of the token
   * @param walletListContract_ The WalletList contract address for access control
   */
  constructor(string memory name_, string memory symbol_, IWalletList walletListContract_) ERC20(name_, symbol_) Ownable(_msgSender()) {
    address walletListAddr = address(walletListContract_);
    if (walletListAddr == address(0) || walletListAddr.code.length == 0) {
      revert InvalidWalletList(walletListAddr);
    }
    walletListContract = walletListContract_;
  }

  /**
   * @notice Modifier to check if the contract is not paused
   * @dev Reverts if the contract is currently paused
   */
  modifier whenNotPaused() {
    if (paused) {
      revert AlreadyPaused();
    }
    _;
  }

  /**
   * @notice Mints new tokens to a specified address
   * @dev Only callable by the contract owner
   * @param to The address to receive the minted tokens
   * @param amount The amount of tokens to mint
   */
  function mint(address to, uint256 amount) external onlyOwner {
    _mint(to, amount);
  }

  /**
   * @notice Mints tokens to multiple addresses in a single transaction
   * @dev Only callable by the contract owner. More gas efficient than multiple mint calls
   * @param to Array of addresses to receive tokens
   * @param amounts Array of token amounts corresponding to each address
   * @custom:throws ArraysLengthMismatch if array lengths don't match
   */
  function batchMint(address[] calldata to, uint256[] calldata amounts) external onlyOwner {
    if (to.length != amounts.length) {
      revert ArraysLengthMismatch();
    }
    for (uint256 i = 0; i < to.length; i++) {
      _mint(to[i], amounts[i]);
    }
  }

  /**
   * @notice Burns tokens from the caller's balance
   * @param amount The amount of tokens to burn
   */
  function burn(uint256 amount) external {
    _burn(_msgSender(), amount);
  }

  /**
   * @notice Burns tokens from multiple addresses in a single transaction
   * @dev Only callable by the contract owner
   * @param from Array of addresses to burn tokens from
   * @param amounts Array of token amounts corresponding to each address
   * @custom:throws ArraysLengthMismatch if array lengths don't match
   */
  function batchBurn(address[] calldata from, uint256[] calldata amounts) external onlyOwner {
    if (from.length != amounts.length) {
      revert ArraysLengthMismatch();
    }
    for (uint256 i = 0; i < from.length; i++) {
      _burn(from[i], amounts[i]);
    }
  }

  /**
   * @notice Sets the pause status of the contract
   * @dev Only callable by the contract owner. When paused, all transfers and approvals are disabled
   * @param status The desired pause status (true = paused, false = unpaused)
   * @custom:throws AlreadyPaused if trying to set the same status
   */
  function pause(bool status) external onlyOwner {
    if (paused == status) {
      revert AlreadyPaused();
    }
    paused = status;
    emit Paused(status);
  }

  /**
   * @notice Checks if an address can transfer tokens
   * @dev An address can transfer if it's whitelisted and not in frozen or blacklists
   * @param account The address to check
   * @return bool True if the address can transfer, false otherwise
   */
  function canTransfer(address account) public view returns (bool) {
    return _isAddressInList(WHITELIST, account, address(0))
        && !_isAddressInList(FROZENLIST, account, address(0))
        && !_isAddressInList(BLACKLIST, account, address(0));
  }

  /**
   * @notice Internal function that handles token transfers with access control
   * @dev Overrides ERC20 _update to enforce whitelist/blacklist/frozenlist checks
   * @param from The address sending tokens (address(0) for minting)
   * @param to The address receiving tokens (address(0) for burning)
   * @param amount The amount of tokens to transfer
   * @custom:throws NotWhitelisted if sender or receiver fails access control checks
   * @custom:security Security note: The owner can transfer tokens even if addresses are frozen or blacklisted, but both sender and receiver must still be whitelisted.
   */
  function _update(address from, address to, uint256 amount) internal virtual override whenNotPaused {
    if (_msgSender() != owner()) {
      if (from != address(0) && !canTransfer(from)) {
        revert NotWhitelisted(from);
      }
      if (to != address(0) && !canTransfer(to)) {
        revert NotWhitelisted(to);
      }
      if (from != address(0) && _msgSender() != from && !canTransfer(_msgSender())) {
        revert NotWhitelisted(_msgSender());
      }
    } else {
      if (from != address(0) && !_isAddressInList(WHITELIST, from, address(0))) {
        revert NotWhitelisted(from);
      }
      if (to != address(0) && !_isAddressInList(WHITELIST, to, address(0))) {
        revert NotWhitelisted(to);
      }
    }
    super._update(from, to, amount);
  }

  /**
   * @notice Approves a spender to spend tokens on behalf of the caller
   * @dev Overrides ERC20 approve to enforce whitelist checks on both parties
   * @param spender The address authorized to spend tokens
   * @param amount The amount of tokens the spender can transfer
   * @return bool True if approval was successful
   * @custom:throws NotWhitelisted if caller or spender is not whitelisted
   */
  function approve(address spender, uint256 amount) public virtual override returns (bool) {
    if (!canTransfer(_msgSender())) {
      revert NotWhitelisted(_msgSender());
    }
    if (amount != 0) {
      if (paused) {
        revert AlreadyPaused();
      }
      if (!canTransfer(spender)) {
        revert NotWhitelisted(spender);
      }
    }
    return super.approve(spender, amount);
  }

  /**
   * @notice Freezes tokens for multiple accounts in a single transaction
   * @dev Transfers tokens to the contract and tracks them in frozenBalance mapping
   * @param accounts Array of account addresses to freeze tokens for
   * @param amounts Array of token amounts to freeze for each account
   * @return bool True if successful
   * @custom:throws ArraysLengthMismatch if array lengths don't match
   * @custom:throws MemberNotRegistered if caller is not authorized to freeze for the account
   * @custom:security Only the owner or the member who whitelisted an account can freeze their tokens
   */
  function batchFreeze(address[] calldata accounts, uint256[] calldata amounts) external virtual returns (bool) {
    if (accounts.length != amounts.length) {
      revert ArraysLengthMismatch();
    }

    for (uint256 i = 0; i < accounts.length; i++) {
      if (!_isAddressInList(WHITELIST, accounts[i], _msgSender()) && _msgSender() != owner()) {
        revert MemberNotRegistered();
      }
      _freeze(accounts[i], amounts[i]);
    }
    return true;
  }

  /**
   * @notice Unfreezes tokens for multiple accounts in a single transaction
   * @dev Returns frozen tokens from the contract back to the account owners
   * @param accounts Array of account addresses to unfreeze tokens for
   * @param amounts Array of token amounts to unfreeze for each account
   * @return bool True if successful
   * @custom:throws ArraysLengthMismatch if array lengths don't match
   * @custom:throws MemberNotRegistered if caller is not authorized to unfreeze for the account
   * @custom:throws InsufficientFrozenBalance if trying to unfreeze more than available
   */
  function batchUnFreeze(address[] calldata accounts, uint256[] calldata amounts) external virtual returns (bool) {
    if (accounts.length != amounts.length) {
      revert ArraysLengthMismatch();
    }

    for (uint256 i = 0; i < accounts.length; i++) {
      if (!_isAddressInList(WHITELIST, accounts[i], _msgSender()) && _msgSender() != owner()) {
        revert MemberNotRegistered();
      }
      _unFreeze(accounts[i], amounts[i]);
    }
    return true;
  }

  /**
   * @notice Internal helper to check if an address is in a specific list
   * @dev Queries the WalletList contract for list membership
   * @param listName The name of the list to check (WHITELIST, FROZENLIST, BLACKLIST, etc.)
   * @param account The address to check
   * @param memberAddress The expected member address (address(0) means check if listed at all)
   * @return bool True if the address is in the list according to the criteria
   */
  function _isAddressInList(bytes32 listName, address account, address memberAddress) internal view returns (bool) {
    if (memberAddress == address(0)) {
      return walletListContract.addressList(listName, account) != memberAddress;
    }
    return walletListContract.addressList(listName, account) == memberAddress;
  }

  /**
   * @notice Internal function to freeze tokens for an account
   * @dev Transfers tokens to the contract and increments frozen balance
   * @param account The account whose tokens are being frozen
   * @param amount The amount of tokens to freeze
   */
  function _freeze(address account, uint256 amount) internal virtual {
    _transfer(account, address(this), amount);
    frozenBalance[account] = frozenBalance[account] + amount;
    emit Freeze(account, amount);
  }

  /**
   * @notice Internal function to unfreeze tokens for an account
   * @dev Transfers frozen tokens back to the owner and decrements frozen balance
   * @param account The account whose tokens are being unfrozen
   * @param amount The amount of tokens to unfreeze
   * @custom:throws InsufficientFrozenBalance if amount exceeds frozen balance
   */
  function _unFreeze(address account, uint256 amount) internal virtual {
    if (frozenBalance[account] < amount) {
      revert InsufficientFrozenBalance();
    }
    _transfer(address(this), account, amount);
    frozenBalance[account] = frozenBalance[account] - amount;
    emit UnFreeze(account, amount);
  }

  /**
   * @notice Adds a document URL (typically IPFS CID) to the token
   * @dev Only callable by the contract owner
   * @param url The IPFS CID or URL to add
   */
  function addDocumentUrl(string calldata url) external onlyOwner {
    for (uint256 i = 0; i < ipfsCids.length; i++) {
      if (keccak256(bytes(ipfsCids[i])) == keccak256(bytes(url))) {
        revert DuplicateUrl();
      }
    }
    ipfsCids.push(url);
    emit DocumentUrlAdded(url);
  }

  /**
   * @notice Removes a document URL at the specified index
   * @dev Only callable by the contract owner. Uses swap-and-pop for gas efficiency
   * @param index The index of the document URL to remove
   * @custom:throws IndexOutOfBounds if index is invalid
   */
  function removeDocumentUrl(uint256 index) external onlyOwner {
    if (index >= ipfsCids.length) {
      revert IndexOutOfBounds();
    }
    string memory removedUrl = ipfsCids[index];
    ipfsCids[index] = ipfsCids[ipfsCids.length - 1];
    ipfsCids.pop();
    emit DocumentUrlRemoved(removedUrl);
  }

  /**
   * @notice Returns all document URLs associated with this token
   * @return string[] Array of IPFS CIDs or URLs
   */
  function getIpfsCids() external view returns (string[] memory) {
    return ipfsCids;
  }

  /**
   * @notice Burns frozen tokens from multiple accounts
   * @dev Only callable by the contract owner. Permanently destroys frozen tokens
   * @param accounts Array of account addresses whose frozen tokens will be burned
   * @param amounts Array of frozen token amounts to burn for each account
   * @custom:throws ArraysLengthMismatch if array lengths don't match
   * @custom:throws InsufficientFrozenBalance if account has insufficient frozen balance
   */
  function batchBurnFrozen(address[] calldata accounts, uint256[] calldata amounts) external onlyOwner {
    if (accounts.length != amounts.length) {
      revert ArraysLengthMismatch();
    }

    for (uint256 i = 0; i < accounts.length; i++) {
      if (frozenBalance[accounts[i]] < amounts[i]) {
        revert InsufficientFrozenBalance();
      }

      frozenBalance[accounts[i]] = frozenBalance[accounts[i]] - amounts[i];
      _burn(address(this), amounts[i]);

      emit FrozenBurned(accounts[i], amounts[i]);
    }
  }
}


// ===== FILE: contracts/fsc20/interfaces/IFSC20.sol =====
// SPDX-License-Identifier: GPL-2.0-or-later
pragma solidity ^0.8.9;

import '@openzeppelin/contracts/token/ERC20/IERC20.sol';
import '../../walletList/interfaces/IWalletList.sol';

/**
 * @title IFSC20
 * @notice Interface for FSC20 token - a compliant ERC20 token with whitelist, freeze, and document management capabilities
 * @dev Extends ERC20 functionality with access control via WalletList integration
 */
interface IFSC20 {
  // Events

  /**
   * @notice Emitted when tokens are frozen for an account
   * @param owner The address whose tokens are being frozen
   * @param value The amount of tokens frozen
   */
  event Freeze(address indexed owner, uint256 value);

  /**
   * @notice Emitted when frozen tokens are unfrozen for an account
   * @param owner The address whose tokens are being unfrozen
   * @param value The amount of tokens unfrozen
   */
  event UnFreeze(address indexed owner, uint256 value);

  /**
   * @notice Emitted when the contract pause status changes
   * @param status The new pause status (true = paused, false = unpaused)
   */
  event Paused(bool status);

  /**
   * @notice Emitted when a new document URL is added to the token
   * @param url The IPFS CID or URL of the document
   */
  event DocumentUrlAdded(string url);

  /**
   * @notice Emitted when a document URL is removed from the token
   * @param url The IPFS CID or URL that was removed
   */
  event DocumentUrlRemoved(string url);

  /**
   * @notice Emitted when frozen tokens are burned
   * @param account The account whose frozen tokens were burned
   * @param amount The amount of frozen tokens burned
   */
  event FrozenBurned(address indexed account, uint256 amount);

  // Custom error definitions

  /**
   * @notice Thrown when attempting to set pause status to its current value
   */
  error AlreadyPaused();

  /**
   * @notice Thrown when array parameters have mismatched lengths
   */
  error ArraysLengthMismatch();

  /**
   * @notice Thrown when an address is not whitelisted or is blacklisted/frozen
   * @param account The address that failed the whitelist check
   */
  error NotWhitelisted(address account);

  /**
   * @notice Thrown when a member is not registered in the system
   */
  error MemberNotRegistered();

  /**
   * @notice Thrown when attempting to unfreeze more tokens than available
   */
  error InsufficientFrozenBalance();

  /**
   * @notice Thrown when accessing an invalid array index
   */
  error IndexOutOfBounds();

  /**
   * @notice Thrown when attempting to add a duplicate document URL
   */
  error DuplicateUrl();

  /**
   * @notice Thrown when the WalletList address provided to the constructor is zero or has no contract code
   * @param walletList The invalid address that was supplied
   */
  error InvalidWalletList(address walletList);

  // External functions

  /**
   * @notice Mints new tokens to a specified address
   * @dev Only callable by contract owner
   * @param to The address to receive the minted tokens
   * @param amount The amount of tokens to mint
   */
  function mint(address to, uint256 amount) external;

  /**
   * @notice Mints tokens to multiple addresses in a single transaction
   * @dev Only callable by contract owner. Arrays must have matching lengths
   * @param to Array of addresses to receive tokens
   * @param amounts Array of amounts corresponding to each address
   */
  function batchMint(address[] calldata to, uint256[] calldata amounts) external;

  /**
   * @notice Burns tokens from the caller's balance
   * @param amount The amount of tokens to burn
   */
  function burn(uint256 amount) external;

  /**
   * @notice Burns tokens from multiple addresses in a single transaction
   * @dev Only callable by contract owner. Arrays must have matching lengths
   * @param from Array of addresses to burn tokens from
   * @param amounts Array of amounts corresponding to each address
   */
  function batchBurn(address[] calldata from, uint256[] calldata amounts) external;

  /**
   * @notice Sets the pause status of the contract
   * @dev Only callable by contract owner. When paused, transfers and approvals are disabled
   * @param status The desired pause status (true = paused, false = unpaused)
   */
  function pause(bool status) external;

  /**
   * @notice Adds a document URL (typically IPFS CID) to the token
   * @dev Only callable by contract owner
   * @param url The IPFS CID or URL to add
   */
  function addDocumentUrl(string calldata url) external;

  /**
   * @notice Removes a document URL at the specified index
   * @dev Only callable by contract owner. Uses swap-and-pop for gas efficiency
   * @param index The index of the document URL to remove
   */
  function removeDocumentUrl(uint256 index) external;

  /**
   * @notice Freezes tokens for multiple accounts
   * @dev Transfers tokens to the contract and tracks frozen balances
   * @param accounts Array of account addresses
   * @param amounts Array of amounts to freeze for each account
   * @return bool True if successful
   */
  function batchFreeze(address[] calldata accounts, uint256[] calldata amounts) external returns (bool);

  /**
   * @notice Unfreezes tokens for multiple accounts
   * @dev Returns frozen tokens from the contract back to the accounts
   * @param accounts Array of account addresses
   * @param amounts Array of amounts to unfreeze for each account
   * @return bool True if successful
   */
  function batchUnFreeze(address[] calldata accounts, uint256[] calldata amounts) external returns (bool);

  /**
   * @notice Burns frozen tokens from multiple accounts
   * @dev Only callable by contract owner. Permanently destroys frozen tokens
   * @param accounts Array of account addresses
   * @param amounts Array of frozen token amounts to burn
   */
  function batchBurnFrozen(address[] calldata accounts, uint256[] calldata amounts) external;

  // View functions

  /**
   * @notice Returns the WalletList contract instance
   * @return IWalletList The WalletList contract used for access control
   */
  function walletListContract() external view returns (IWalletList);

  /**
   * @notice Returns the current pause status
   * @return bool True if contract is paused, false otherwise
   */
  function paused() external view returns (bool);

  /**
   * @notice Returns the frozen balance for a specific account
   * @param account The address to query
   * @return uint256 The amount of frozen tokens
   */
  function frozenBalance(address account) external view returns (uint256);

  /**
   * @notice Returns all document URLs associated with this token
   * @return string[] Array of IPFS CIDs or URLs
   */
  function getIpfsCids() external view returns (string[] memory);
}


// ===== FILE: contracts/walletList/interfaces/IWalletList.sol =====
// SPDX-License-Identifier: GPL-2.0-or-later
pragma solidity ^0.8.0;

/**
 * @title IWalletList
 * @notice Interface for managing multiple address lists with role-based access control
 * @dev Provides a flexible system for managing whitelists, blacklists, frozen lists, and member lists
 *
 * Key Concepts:
 * - Lists: Named collections of addresses (e.g., WHITELIST, FROZENLIST, BLACKLIST, MEMBER)
 * - Roles: Permissions that control who can manage specific lists
 * - Members: Addresses that track which member added a user to a list
 *
 * Common Usage:
 * - MEMBER list: Tracks members who can manage other lists
 * - WHITELIST: Addresses allowed to participate
 * - FROZENLIST: Addresses temporarily restricted
 * - BLACKLIST: Addresses permanently blocked
 */
interface IWalletList {
    // State getters

    /**
     * @notice Returns the member address that added a user to a specific list
     * @param listName The name of the list to query
     * @param user The user address to check
     * @return address The member address that added the user (address(0) if not listed)
     */
    function addressList(bytes32 listName, address user) external view returns (address);

    /**
     * @notice Checks if a role is allowed to manage a specific list
     * @param listName The name of the list to query
     * @param role The role to check
     * @return bool True if the role can manage the list, false otherwise
     */
    function roleManageList(bytes32 listName, bytes32 role) external view returns (bool);

    // List management functions

    /**
     * @notice Adds a user to a specific list
     * @dev Caller must have the required role permission for the list
     * @param listName The name of the list
     * @param role The role required to manage this list
     * @param user The user address to add
     * @custom:throws NoPermission if caller lacks permission
     * @custom:throws AlreadyListed if user is already in the list
     */
    function addToList(bytes32 listName, bytes32 role, address user) external;

    /**
     * @notice Removes a user from a specific list
     * @dev Caller must be the member who originally added the user
     * @param listName The name of the list
     * @param role The role required to manage this list
     * @param user The user address to remove
     * @custom:throws NoPermission if caller lacks permission or didn't add the user
     * @custom:throws NotListed if user is not in the list
     */
    function removeFromList(bytes32 listName, bytes32 role, address user) external;

    /**
     * @notice Adds multiple users to a specific list in a single transaction
     * @dev More gas efficient than multiple addToList calls
     * @param listName The name of the list
     * @param role The role required to manage this list
     * @param users Array of user addresses to add
     */
    function batchAddToList(bytes32 listName, bytes32 role, address[] calldata users) external;

    /**
     * @notice Removes multiple users from a specific list in a single transaction
     * @dev More gas efficient than multiple removeFromList calls
     * @param listName The name of the list
     * @param role The role required to manage this list
     * @param users Array of user addresses to remove
     */
    function batchRemoveFromList(bytes32 listName, bytes32 role, address[] calldata users) external;

    /**
     * @notice Owner-only function to add users on behalf of a specific member
     * @dev Only callable by DEFAULT_ADMIN_ROLE. Useful for migrations or admin operations
     * @param listName The name of the list
     * @param role The role required to manage this list
     * @param users Array of user addresses to add
     * @param member The member address to assign as the manager
     */
    function batchAddToListByAdmin(bytes32 listName, bytes32 role, address[] calldata users, address member) external;

    /**
     * @notice Owner-only function to remove users on behalf of a specific member
     * @dev Only callable by DEFAULT_ADMIN_ROLE. Useful for migrations or admin operations
     * @param listName The name of the list
     * @param role The role required to manage this list
     * @param users Array of user addresses to remove
     * @param member The member address that originally added the users
     */
    function batchRemoveFromListByAdmin(bytes32 listName, bytes32 role, address[] calldata users, address member) external;

    /**
     * @notice Checks if an address is in a specific list
     * @param listName The name of the list to check
     * @param account The address to verify
     * @return bool True if the address is in the list, false otherwise
     */
    function isAddressInList(bytes32 listName, address account) external view returns (bool);

    // Role management functions

    /**
     * @notice Sets whether a role is allowed to manage a specific list
     * @dev Only callable by DEFAULT_ADMIN_ROLE
     * @param listName The name of the list
     * @param role The role to grant or revoke permission
     * @param allowed True to allow, false to disallow
     */
    function setRoleManageList(bytes32 listName, bytes32 role, bool allowed) external;

    // Events

    /**
     * @notice Emitted when a user is added to or removed from a list
     * @param listName The name of the list that was modified
     * @param user The user address that was added or removed
     * @param member The member address that performed the action (address(0) for removal)
     */
    event ListChanged(bytes32 indexed listName, address indexed user, address indexed member);

    /**
     * @notice Emitted when list permissions are changed for a role
     * @param listName The name of the list
     * @param role The role whose permissions were changed
     * @param allowed The new permission status
     */
    event ListPermissionChanged(bytes32 indexed listName, bytes32 indexed role, bool allowed);

    // Errors

    /**
     * @notice Thrown when attempting to add a user who is already in the list
     * @param user The address that was already listed
     */
    error AlreadyListed(address user);

    /**
     * @notice Thrown when attempting to remove a user who is not in the list
     * @param user The address that was not listed
     */
    error NotListed(address user);

    /**
     * @notice Thrown when caller lacks permission for the requested operation
     * @param user The address that lacked permission
     */
    error NoPermission(address user);

    /**
     * @notice Thrown when attempting to set a role permission that is already set
     */
    error NoChange();
}

// ===== FILE: contracts/walletList/WalletList.sol =====
// SPDX-License-Identifier: GPL-2.0-or-later
pragma solidity ^0.8.9;

import './interfaces/IWalletList.sol';
import '@openzeppelin/contracts/access/Ownable2Step.sol';

/**
 * @title WalletList
 * @notice Manages multiple address lists with role-based access control for token compliance
 * @dev Implements IWalletList using addressList as single source of truth for all permissions
 *
 * Architecture:
 * - addressList is the sole permission system (no OZ AccessControl dependency)
 * - Maintains multiple named lists (WHITELIST, FROZENLIST, BLACKLIST, MEMBER)
 * - Tracks which member added each user to a list for accountability
 * - Supports batch operations for gas efficiency
 *
 * Access Control Hierarchy:
 * 1. DEFAULT_ADMIN_ROLE: Addresses in addressList[DEFAULT_ADMIN_ROLE] can perform admin operations
 * 2. MEMBER: Can add/remove users to WHITELIST and FROZENLIST
 * 3. Users: Addresses managed in various lists
 *
 * Data Structure:
 * - addressList[listName][userAddress] => memberAddress
 *   Maps a user in a specific list to the member who added them
 *   address(0) means not in the list
 *
 * - roleManageList[listName][role] => bool
 *   Tracks which roles can manage which lists
 */
contract WalletList is IWalletList, Ownable2Step {
  /// @notice Admin role identifier (matches OZ DEFAULT_ADMIN_ROLE = 0x00 for interface compatibility)
  bytes32 public constant DEFAULT_ADMIN_ROLE = 0x00;

  /// @notice Role identifier for members who can manage whitelists and frozen lists
  bytes32 public constant MEMBER = keccak256('MEMBER');

  /// @notice List identifier for whitelisted addresses
  bytes32 public constant WHITELIST = keccak256('WHITELIST');

  /// @notice List identifier for frozen addresses
  bytes32 public constant FROZENLIST = keccak256('FROZENLIST');

  /**
   * @notice Maps list names and user addresses to the member who added them
   * @dev Structure: listName => userAddress => memberAddress
   * Examples:
   * - WHITELIST -> ClientAddress => MemberAddress (member who whitelisted the client)
   * - FROZENLIST -> ClientAddress => MemberAddress (member who froze the client)
   * - MEMBER -> MemberAddress => AdminAddress (admin who added the member)
   * - BLACKLIST -> ClientAddress => AdminAddress (admin who blacklisted the client)
   */
  mapping(bytes32 => mapping(address => address)) public addressList;

  /**
   * @notice Maps list names and roles to permission status
   * @dev Structure: listName => role => allowed
   * Examples:
   * - WHITELIST -> MEMBER -> true (members can manage whitelist)
   * - FROZENLIST -> MEMBER -> true (members can manage frozen list)
   */
  mapping(bytes32 => mapping(bytes32 => bool)) public roleManageList;

  error NotAdmin(address account);

  modifier onlyAdmin() {
    if (!isAdmin(msg.sender)) revert NotAdmin(msg.sender);
    _;
  }

  /**
   * @notice Checks if an account has admin privileges
   * @param account The address to check
   * @return True if the account is in the DEFAULT_ADMIN_ROLE list
   */
  function isAdmin(address account) public view returns (bool) {
    return isAddressInList(DEFAULT_ADMIN_ROLE, account);
  }

  /**
   * @notice Initializes the WalletList contract with default permissions
   * @dev Sets up the deployer as admin and configures initial role permissions
   */
  constructor() Ownable(_msgSender()) {
    _setAddressList(DEFAULT_ADMIN_ROLE, msg.sender, msg.sender);
    _setRoleManageList(MEMBER, DEFAULT_ADMIN_ROLE, true);
    _setRoleManageList(WHITELIST, MEMBER, true);
    _setRoleManageList(FROZENLIST, MEMBER, true);
  }

  // #region LIST FUNCTIONS

  /**
   * @notice Adds a user to a specific list
   * @dev Caller must have permission to manage the list. Delegates to _listAction
   * @param listName The name of the list (e.g., WHITELIST, FROZENLIST)
   * @param role The role required to manage this list (e.g., MEMBER)
   * @param user The address to add to the list
   */
  function addToList(bytes32 listName, bytes32 role, address user) external {
    _listAction(listName, role, user, msg.sender, true);
  }

  /**
   * @notice Removes a user from a specific list
   * @dev Caller must be the member who originally added the user. Delegates to _listAction
   * @param listName The name of the list (e.g., WHITELIST, FROZENLIST)
   * @param role The role required to manage this list (e.g., MEMBER)
   * @param user The address to remove from the list
   */
  function removeFromList(bytes32 listName, bytes32 role, address user) external {
    _listAction(listName, role, user, msg.sender, false);
  }

  /**
   * @notice Adds multiple users to a list in a single transaction
   * @dev More gas efficient than multiple addToList calls
   * @param listName The name of the list
   * @param role The role required to manage this list
   * @param users Array of addresses to add
   */
  function batchAddToList(bytes32 listName, bytes32 role, address[] calldata users) external {
    _batchListAction(listName, role, users, msg.sender, true);
  }

  /**
   * @notice Removes multiple users from a list in a single transaction
   * @dev More gas efficient than multiple removeFromList calls
   * @param listName The name of the list
   * @param role The role required to manage this list
   * @param users Array of addresses to remove
   */
  function batchRemoveFromList(bytes32 listName, bytes32 role, address[] calldata users) external {
    _batchListAction(listName, role, users, msg.sender, false);
  }

  /**
   * @notice Admin function to add users on behalf of a specific manager
   * @dev Only callable by DEFAULT_ADMIN_ROLE. Useful for migrations or bulk operations
   * @param listName The name of the list
   * @param role The role required to manage this list
   * @param users Array of addresses to add
   * @param manager The member address to assign as the one who added these users
   */
  function batchAddToListByAdmin(bytes32 listName, bytes32 role, address[] calldata users, address manager) external onlyAdmin {
    require(listName != DEFAULT_ADMIN_ROLE, "Use addAdmin for admin list");
    _batchListAction(listName, role, users, manager, true);
  }

  /**
   * @notice Admin function to force-remove users from a list, bypassing ownership check
   * @dev Only callable by admin. Resolves the deadlock where users would otherwise be
   *      stuck in a list after their original manager has been revoked. The `role` and
   *      `manager` params are retained for interface compatibility but not used.
   * @param listName The name of the list
   * @param users Array of addresses to remove
   */
  function batchRemoveFromListByAdmin(bytes32 listName, bytes32, address[] calldata users, address) external onlyAdmin {
    require(listName != DEFAULT_ADMIN_ROLE, "Use removeAdmin for admin list");
    for (uint256 i = 0; i < users.length; i++) {
      if (addressList[listName][users[i]] == address(0)) revert NotListed(users[i]);
      _setAddressList(listName, users[i], address(0));
    }
  }

  /**
   * @notice Checks if an address is present in a specific list
   * @param listName The name of the list to check
   * @param account The address to verify
   * @return bool True if the address is in the list (managed by any member), false otherwise
   */
  function isAddressInList(bytes32 listName, address account) public view returns (bool) {
    address manager = addressList[listName][account];
    return manager != address(0);
  }
  // #endregion

  /**
   * @notice Internal function to add or remove a user from a list
   * @dev Performs permission checks and validates state transitions
   * @param listName The name of the list
   * @param role The role required to manage this list
   * @param user The user address to add or remove
   * @param manager The member address performing the action
   * @param add True to add, false to remove
   * @custom:throws NoPermission if manager lacks permission
   * @custom:throws AlreadyListed if adding a user who's already in the list
   * @custom:throws NotListed if removing a user who's not in the list
   */
  function _listAction(bytes32 listName, bytes32 role, address user, address manager, bool add) internal {
    _checkRolePermission(listName, role, manager);
    require(user != address(0), 'Invalid address');
    if (add) {
      if (addressList[listName][user] != address(0)) {
        revert AlreadyListed(user);
      }
      _setAddressList(listName, user, manager);
    } else {
      if (addressList[listName][user] == address(0)) {
        revert NotListed(user);
      }
      if (addressList[listName][user] != manager) {
        revert NoPermission(user);
      }
      _setAddressList(listName, user, address(0));
    }
  }

  /**
   * @notice Internal function to update the addressList mapping and emit event
   * @dev Sets the manager for a user in a specific list
   * @param listName The name of the list
   * @param user The user address
   * @param manager The member address (address(0) for removal)
   */
  function _setAddressList(bytes32 listName, address user, address manager) internal {
    addressList[listName][user] = manager;
    emit ListChanged(listName, user, manager);
  }

  /**
   * @notice Internal function to perform batch list operations
   * @dev Iterates through users array and calls _listAction for each
   * @param listName The name of the list
   * @param role The role required to manage this list
   * @param users Array of user addresses
   * @param manager The member address performing the actions
   * @param add True to add users, false to remove
   */
  function _batchListAction(bytes32 listName, bytes32 role, address[] calldata users, address manager, bool add) internal {
    for (uint256 i = 0; i < users.length; i++) {
      _listAction(listName, role, users[i], manager, add);
    }
  }

  /**
   * @notice Sets whether a role is allowed to manage a specific list
   * @dev Only callable by DEFAULT_ADMIN_ROLE
   * @param listName The name of the list
   * @param role The role to grant or revoke permission
   * @param allowed True to allow, false to disallow
   */
  /// @notice Adds an address to the admin list
  /// @dev Only callable by the contract owner (Ownable2Step)
  /// @param account The address to grant admin role
  function addAdmin(address account) external onlyOwner {
    require(account != address(0), "Invalid address");
    if (addressList[DEFAULT_ADMIN_ROLE][account] != address(0)) revert AlreadyListed(account);
    _setAddressList(DEFAULT_ADMIN_ROLE, account, msg.sender);
  }

  /// @notice Removes an address from the admin list
  /// @dev Only callable by the contract owner (Ownable2Step)
  /// @param account The address to revoke admin role
  function removeAdmin(address account) external onlyOwner {
    if (addressList[DEFAULT_ADMIN_ROLE][account] == address(0)) revert NotListed(account);
    _setAddressList(DEFAULT_ADMIN_ROLE, account, address(0));
  }

  function setRoleManageList(bytes32 listName, bytes32 role, bool allowed) external onlyAdmin {
    require(listName != DEFAULT_ADMIN_ROLE, "Admin list managed by owner only");
    _setRoleManageList(listName, role, allowed);
  }

  /**
   * @notice Internal function to update role permissions and emit event
   * @param listName The name of the list
   * @param role The role being granted or revoked permission
   * @param allowed The new permission status
   */
  function _setRoleManageList(bytes32 listName, bytes32 role, bool allowed) internal {
    if (roleManageList[listName][role] == allowed) revert NoChange();
    roleManageList[listName][role] = allowed;
    emit ListPermissionChanged(listName, role, allowed);
  }

  /**
   * @notice Internal function to verify a manager has permission to manage a list
   * @dev Checks both role permission and manager's presence in the role list
   * @param listName The name of the list to manage
   * @param role The role required to manage the list
   * @param manager The address attempting to manage the list
   * @custom:throws NoPermission if role is not allowed or manager doesn't have the role
   */
  function _checkRolePermission(bytes32 listName, bytes32 role, address manager) internal view {
    if (!roleManageList[listName][role] || !isAddressInList(role, manager)) {
      revert NoPermission(manager);
    }
  }
}
