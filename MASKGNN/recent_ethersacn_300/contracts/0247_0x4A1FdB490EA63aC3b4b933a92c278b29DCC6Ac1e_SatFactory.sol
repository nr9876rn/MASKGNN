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


// ===== FILE: _openzeppelin/contracts/token/ERC721/IERC721.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (token/ERC721/IERC721.sol)

pragma solidity >=0.6.2;

import {IERC165} from "../../utils/introspection/IERC165.sol";

/**
 * @dev Required interface of an ERC-721 compliant contract.
 */
interface IERC721 is IERC165 {
    /**
     * @dev Emitted when `tokenId` token is transferred from `from` to `to`.
     */
    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);

    /**
     * @dev Emitted when `owner` enables `approved` to manage the `tokenId` token.
     */
    event Approval(address indexed owner, address indexed approved, uint256 indexed tokenId);

    /**
     * @dev Emitted when `owner` enables or disables (`approved`) `operator` to manage all of its assets.
     */
    event ApprovalForAll(address indexed owner, address indexed operator, bool approved);

    /**
     * @dev Returns the number of tokens in ``owner``'s account.
     */
    function balanceOf(address owner) external view returns (uint256 balance);

    /**
     * @dev Returns the owner of the `tokenId` token.
     *
     * Requirements:
     *
     * - `tokenId` must exist.
     */
    function ownerOf(uint256 tokenId) external view returns (address owner);

    /**
     * @dev Safely transfers `tokenId` token from `from` to `to`.
     *
     * Requirements:
     *
     * - `from` cannot be the zero address.
     * - `to` cannot be the zero address.
     * - `tokenId` token must exist and be owned by `from`.
     * - If the caller is not `from`, it must be approved to move this token by either {approve} or {setApprovalForAll}.
     * - If `to` refers to a smart contract, it must implement {IERC721Receiver-onERC721Received}, which is called upon
     *   a safe transfer.
     *
     * Emits a {Transfer} event.
     */
    function safeTransferFrom(address from, address to, uint256 tokenId, bytes calldata data) external;

    /**
     * @dev Safely transfers `tokenId` token from `from` to `to`, checking first that contract recipients
     * are aware of the ERC-721 protocol to prevent tokens from being forever locked.
     *
     * Requirements:
     *
     * - `from` cannot be the zero address.
     * - `to` cannot be the zero address.
     * - `tokenId` token must exist and be owned by `from`.
     * - If the caller is not `from`, it must have been allowed to move this token by either {approve} or
     *   {setApprovalForAll}.
     * - If `to` refers to a smart contract, it must implement {IERC721Receiver-onERC721Received}, which is called upon
     *   a safe transfer.
     *
     * Emits a {Transfer} event.
     */
    function safeTransferFrom(address from, address to, uint256 tokenId) external;

    /**
     * @dev Transfers `tokenId` token from `from` to `to`.
     *
     * WARNING: Note that the caller is responsible to confirm that the recipient is capable of receiving ERC-721
     * or else they may be permanently lost. Usage of {safeTransferFrom} prevents loss, though the caller must
     * understand this adds an external call which potentially creates a reentrancy vulnerability.
     *
     * Requirements:
     *
     * - `from` cannot be the zero address.
     * - `to` cannot be the zero address.
     * - `tokenId` token must be owned by `from`.
     * - If the caller is not `from`, it must be approved to move this token by either {approve} or {setApprovalForAll}.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(address from, address to, uint256 tokenId) external;

    /**
     * @dev Gives permission to `to` to transfer `tokenId` token to another account.
     * The approval is cleared when the token is transferred.
     *
     * Only a single account can be approved at a time, so approving the zero address clears previous approvals.
     *
     * Requirements:
     *
     * - The caller must own the token or be an approved operator.
     * - `tokenId` must exist.
     *
     * Emits an {Approval} event.
     */
    function approve(address to, uint256 tokenId) external;

    /**
     * @dev Approve or remove `operator` as an operator for the caller.
     * Operators can call {transferFrom} or {safeTransferFrom} for any token owned by the caller.
     *
     * Requirements:
     *
     * - The `operator` cannot be the address zero.
     *
     * Emits an {ApprovalForAll} event.
     */
    function setApprovalForAll(address operator, bool approved) external;

    /**
     * @dev Returns the account approved for `tokenId` token.
     *
     * Requirements:
     *
     * - `tokenId` must exist.
     */
    function getApproved(uint256 tokenId) external view returns (address operator);

    /**
     * @dev Returns if the `operator` is allowed to manage all of the assets of `owner`.
     *
     * See {setApprovalForAll}
     */
    function isApprovedForAll(address owner, address operator) external view returns (bool);
}


// ===== FILE: _openzeppelin/contracts/token/ERC721/IERC721Receiver.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (token/ERC721/IERC721Receiver.sol)

pragma solidity >=0.5.0;

/**
 * @title ERC-721 token receiver interface
 * @dev Interface for any contract that wants to support safeTransfers
 * from ERC-721 asset contracts.
 */
interface IERC721Receiver {
    /**
     * @dev Whenever an {IERC721} `tokenId` token is transferred to this contract via {IERC721-safeTransferFrom}
     * by `operator` from `from`, this function is called.
     *
     * It must return its Solidity selector to confirm the token transfer.
     * If any other value is returned or the interface is not implemented by the recipient, the transfer will be
     * reverted.
     *
     * The selector can be obtained in Solidity with `IERC721Receiver.onERC721Received.selector`.
     */
    function onERC721Received(
        address operator,
        address from,
        uint256 tokenId,
        bytes calldata data
    ) external returns (bytes4);
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


// ===== FILE: _openzeppelin/contracts/utils/introspection/IERC165.sol =====
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


// ===== FILE: _openzeppelin/contracts/utils/ReentrancyGuard.sol =====
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


// ===== FILE: _openzeppelin/contracts/utils/StorageSlot.sol =====
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


// ===== FILE: _uniswap/v4-core/src/interfaces/callback/IUnlockCallback.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @notice Interface for the callback executed when an address unlocks the pool manager
interface IUnlockCallback {
    /// @notice Called by the pool manager on `msg.sender` when the manager is unlocked
    /// @param data The data that was passed to the call to unlock
    /// @return Any data that you want to be returned from the unlock call
    function unlockCallback(bytes calldata data) external returns (bytes memory);
}


// ===== FILE: _uniswap/v4-core/src/interfaces/external/IERC20Minimal.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title Minimal ERC20 interface for Uniswap
/// @notice Contains a subset of the full ERC20 interface that is used in Uniswap V3
interface IERC20Minimal {
    /// @notice Returns an account's balance in the token
    /// @param account The account for which to look up the number of tokens it has, i.e. its balance
    /// @return The number of tokens held by the account
    function balanceOf(address account) external view returns (uint256);

    /// @notice Transfers the amount of token from the `msg.sender` to the recipient
    /// @param recipient The account that will receive the amount transferred
    /// @param amount The number of tokens to send from the sender to the recipient
    /// @return Returns true for a successful transfer, false for an unsuccessful transfer
    function transfer(address recipient, uint256 amount) external returns (bool);

    /// @notice Returns the current allowance given to a spender by an owner
    /// @param owner The account of the token owner
    /// @param spender The account of the token spender
    /// @return The current allowance granted by `owner` to `spender`
    function allowance(address owner, address spender) external view returns (uint256);

    /// @notice Sets the allowance of a spender from the `msg.sender` to the value `amount`
    /// @param spender The account which will be allowed to spend a given amount of the owners tokens
    /// @param amount The amount of tokens allowed to be used by `spender`
    /// @return Returns true for a successful approval, false for unsuccessful
    function approve(address spender, uint256 amount) external returns (bool);

    /// @notice Transfers `amount` tokens from `sender` to `recipient` up to the allowance given to the `msg.sender`
    /// @param sender The account from which the transfer will be initiated
    /// @param recipient The recipient of the transfer
    /// @param amount The amount of the transfer
    /// @return Returns true for a successful transfer, false for unsuccessful
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);

    /// @notice Event emitted when tokens are transferred from one address to another, either via `#transfer` or `#transferFrom`.
    /// @param from The account from which the tokens were sent, i.e. the balance decreased
    /// @param to The account to which the tokens were sent, i.e. the balance increased
    /// @param value The amount of tokens that were transferred
    event Transfer(address indexed from, address indexed to, uint256 value);

    /// @notice Event emitted when the approval amount for the spender of a given owner's tokens changes.
    /// @param owner The account that approved spending of its tokens
    /// @param spender The account for which the spending allowance was modified
    /// @param value The new allowance from the owner to the spender
    event Approval(address indexed owner, address indexed spender, uint256 value);
}


// ===== FILE: _uniswap/v4-core/src/interfaces/IHooks.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {PoolKey} from "../types/PoolKey.sol";
import {BalanceDelta} from "../types/BalanceDelta.sol";
import {ModifyLiquidityParams, SwapParams} from "../types/PoolOperation.sol";
import {BeforeSwapDelta} from "../types/BeforeSwapDelta.sol";

/// @notice V4 decides whether to invoke specific hooks by inspecting the least significant bits
/// of the address that the hooks contract is deployed to.
/// For example, a hooks contract deployed to address: 0x0000000000000000000000000000000000002400
/// has the lowest bits '10 0100 0000 0000' which would cause the 'before initialize' and 'after add liquidity' hooks to be used.
/// See the Hooks library for the full spec.
/// @dev Should only be callable by the v4 PoolManager.
interface IHooks {
    /// @notice The hook called before the state of a pool is initialized
    /// @param sender The initial msg.sender for the initialize call
    /// @param key The key for the pool being initialized
    /// @param sqrtPriceX96 The sqrt(price) of the pool as a Q64.96
    /// @return bytes4 The function selector for the hook
    function beforeInitialize(address sender, PoolKey calldata key, uint160 sqrtPriceX96) external returns (bytes4);

    /// @notice The hook called after the state of a pool is initialized
    /// @param sender The initial msg.sender for the initialize call
    /// @param key The key for the pool being initialized
    /// @param sqrtPriceX96 The sqrt(price) of the pool as a Q64.96
    /// @param tick The current tick after the state of a pool is initialized
    /// @return bytes4 The function selector for the hook
    function afterInitialize(address sender, PoolKey calldata key, uint160 sqrtPriceX96, int24 tick)
        external
        returns (bytes4);

    /// @notice The hook called before liquidity is added
    /// @param sender The initial msg.sender for the add liquidity call
    /// @param key The key for the pool
    /// @param params The parameters for adding liquidity
    /// @param hookData Arbitrary data handed into the PoolManager by the liquidity provider to be passed on to the hook
    /// @return bytes4 The function selector for the hook
    function beforeAddLiquidity(
        address sender,
        PoolKey calldata key,
        ModifyLiquidityParams calldata params,
        bytes calldata hookData
    ) external returns (bytes4);

    /// @notice The hook called after liquidity is added
    /// @param sender The initial msg.sender for the add liquidity call
    /// @param key The key for the pool
    /// @param params The parameters for adding liquidity
    /// @param delta The caller's balance delta after adding liquidity; the sum of principal delta, fees accrued, and hook delta
    /// @param feesAccrued The fees accrued since the last time fees were collected from this position
    /// @param hookData Arbitrary data handed into the PoolManager by the liquidity provider to be passed on to the hook
    /// @return bytes4 The function selector for the hook
    /// @return BalanceDelta The hook's delta in token0 and token1. Positive: the hook is owed/took currency, negative: the hook owes/sent currency
    function afterAddLiquidity(
        address sender,
        PoolKey calldata key,
        ModifyLiquidityParams calldata params,
        BalanceDelta delta,
        BalanceDelta feesAccrued,
        bytes calldata hookData
    ) external returns (bytes4, BalanceDelta);

    /// @notice The hook called before liquidity is removed
    /// @param sender The initial msg.sender for the remove liquidity call
    /// @param key The key for the pool
    /// @param params The parameters for removing liquidity
    /// @param hookData Arbitrary data handed into the PoolManager by the liquidity provider to be be passed on to the hook
    /// @return bytes4 The function selector for the hook
    function beforeRemoveLiquidity(
        address sender,
        PoolKey calldata key,
        ModifyLiquidityParams calldata params,
        bytes calldata hookData
    ) external returns (bytes4);

    /// @notice The hook called after liquidity is removed
    /// @param sender The initial msg.sender for the remove liquidity call
    /// @param key The key for the pool
    /// @param params The parameters for removing liquidity
    /// @param delta The caller's balance delta after removing liquidity; the sum of principal delta, fees accrued, and hook delta
    /// @param feesAccrued The fees accrued since the last time fees were collected from this position
    /// @param hookData Arbitrary data handed into the PoolManager by the liquidity provider to be be passed on to the hook
    /// @return bytes4 The function selector for the hook
    /// @return BalanceDelta The hook's delta in token0 and token1. Positive: the hook is owed/took currency, negative: the hook owes/sent currency
    function afterRemoveLiquidity(
        address sender,
        PoolKey calldata key,
        ModifyLiquidityParams calldata params,
        BalanceDelta delta,
        BalanceDelta feesAccrued,
        bytes calldata hookData
    ) external returns (bytes4, BalanceDelta);

    /// @notice The hook called before a swap
    /// @param sender The initial msg.sender for the swap call
    /// @param key The key for the pool
    /// @param params The parameters for the swap
    /// @param hookData Arbitrary data handed into the PoolManager by the swapper to be be passed on to the hook
    /// @return bytes4 The function selector for the hook
    /// @return BeforeSwapDelta The hook's delta in specified and unspecified currencies. Positive: the hook is owed/took currency, negative: the hook owes/sent currency
    /// @return uint24 Optionally override the lp fee, only used if three conditions are met: 1. the Pool has a dynamic fee, 2. the value's 2nd highest bit is set (23rd bit, 0x400000), and 3. the value is less than or equal to the maximum fee (1 million)
    function beforeSwap(address sender, PoolKey calldata key, SwapParams calldata params, bytes calldata hookData)
        external
        returns (bytes4, BeforeSwapDelta, uint24);

    /// @notice The hook called after a swap
    /// @param sender The initial msg.sender for the swap call
    /// @param key The key for the pool
    /// @param params The parameters for the swap
    /// @param delta The amount owed to the caller (positive) or owed to the pool (negative)
    /// @param hookData Arbitrary data handed into the PoolManager by the swapper to be be passed on to the hook
    /// @return bytes4 The function selector for the hook
    /// @return int128 The hook's delta in unspecified currency. Positive: the hook is owed/took currency, negative: the hook owes/sent currency
    function afterSwap(
        address sender,
        PoolKey calldata key,
        SwapParams calldata params,
        BalanceDelta delta,
        bytes calldata hookData
    ) external returns (bytes4, int128);

    /// @notice The hook called before donate
    /// @param sender The initial msg.sender for the donate call
    /// @param key The key for the pool
    /// @param amount0 The amount of token0 being donated
    /// @param amount1 The amount of token1 being donated
    /// @param hookData Arbitrary data handed into the PoolManager by the donor to be be passed on to the hook
    /// @return bytes4 The function selector for the hook
    function beforeDonate(
        address sender,
        PoolKey calldata key,
        uint256 amount0,
        uint256 amount1,
        bytes calldata hookData
    ) external returns (bytes4);

    /// @notice The hook called after donate
    /// @param sender The initial msg.sender for the donate call
    /// @param key The key for the pool
    /// @param amount0 The amount of token0 being donated
    /// @param amount1 The amount of token1 being donated
    /// @param hookData Arbitrary data handed into the PoolManager by the donor to be be passed on to the hook
    /// @return bytes4 The function selector for the hook
    function afterDonate(
        address sender,
        PoolKey calldata key,
        uint256 amount0,
        uint256 amount1,
        bytes calldata hookData
    ) external returns (bytes4);
}


// ===== FILE: _uniswap/v4-core/src/libraries/BitMath.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title BitMath
/// @dev This library provides functionality for computing bit properties of an unsigned integer
/// @author Solady (https://github.com/Vectorized/solady/blob/8200a70e8dc2a77ecb074fc2e99a2a0d36547522/src/utils/LibBit.sol)
library BitMath {
    /// @notice Returns the index of the most significant bit of the number,
    ///     where the least significant bit is at index 0 and the most significant bit is at index 255
    /// @param x the value for which to compute the most significant bit, must be greater than 0
    /// @return r the index of the most significant bit
    function mostSignificantBit(uint256 x) internal pure returns (uint8 r) {
        require(x > 0);

        assembly ("memory-safe") {
            r := shl(7, lt(0xffffffffffffffffffffffffffffffff, x))
            r := or(r, shl(6, lt(0xffffffffffffffff, shr(r, x))))
            r := or(r, shl(5, lt(0xffffffff, shr(r, x))))
            r := or(r, shl(4, lt(0xffff, shr(r, x))))
            r := or(r, shl(3, lt(0xff, shr(r, x))))
            // forgefmt: disable-next-item
            r := or(r, byte(and(0x1f, shr(shr(r, x), 0x8421084210842108cc6318c6db6d54be)),
                0x0706060506020500060203020504000106050205030304010505030400000000))
        }
    }

    /// @notice Returns the index of the least significant bit of the number,
    ///     where the least significant bit is at index 0 and the most significant bit is at index 255
    /// @param x the value for which to compute the least significant bit, must be greater than 0
    /// @return r the index of the least significant bit
    function leastSignificantBit(uint256 x) internal pure returns (uint8 r) {
        require(x > 0);

        assembly ("memory-safe") {
            // Isolate the least significant bit.
            x := and(x, sub(0, x))
            // For the upper 3 bits of the result, use a De Bruijn-like lookup.
            // Credit to adhusson: https://blog.adhusson.com/cheap-find-first-set-evm/
            // forgefmt: disable-next-item
            r := shl(5, shr(252, shl(shl(2, shr(250, mul(x,
                0xb6db6db6ddddddddd34d34d349249249210842108c6318c639ce739cffffffff))),
                0x8040405543005266443200005020610674053026020000107506200176117077)))
            // For the lower 5 bits of the result, use a De Bruijn lookup.
            // forgefmt: disable-next-item
            r := or(r, byte(and(div(0xd76453e0, shr(r, x)), 0x1f),
                0x001f0d1e100c1d070f090b19131c1706010e11080a1a141802121b1503160405))
        }
    }
}


// ===== FILE: _uniswap/v4-core/src/libraries/CustomRevert.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title Library for reverting with custom errors efficiently
/// @notice Contains functions for reverting with custom errors with different argument types efficiently
/// @dev To use this library, declare `using CustomRevert for bytes4;` and replace `revert CustomError()` with
/// `CustomError.selector.revertWith()`
/// @dev The functions may tamper with the free memory pointer but it is fine since the call context is exited immediately
library CustomRevert {
    /// @dev ERC-7751 error for wrapping bubbled up reverts
    error WrappedError(address target, bytes4 selector, bytes reason, bytes details);

    /// @dev Reverts with the selector of a custom error in the scratch space
    function revertWith(bytes4 selector) internal pure {
        assembly ("memory-safe") {
            mstore(0, selector)
            revert(0, 0x04)
        }
    }

    /// @dev Reverts with a custom error with an address argument in the scratch space
    function revertWith(bytes4 selector, address addr) internal pure {
        assembly ("memory-safe") {
            mstore(0, selector)
            mstore(0x04, and(addr, 0xffffffffffffffffffffffffffffffffffffffff))
            revert(0, 0x24)
        }
    }

    /// @dev Reverts with a custom error with an int24 argument in the scratch space
    function revertWith(bytes4 selector, int24 value) internal pure {
        assembly ("memory-safe") {
            mstore(0, selector)
            mstore(0x04, signextend(2, value))
            revert(0, 0x24)
        }
    }

    /// @dev Reverts with a custom error with a uint160 argument in the scratch space
    function revertWith(bytes4 selector, uint160 value) internal pure {
        assembly ("memory-safe") {
            mstore(0, selector)
            mstore(0x04, and(value, 0xffffffffffffffffffffffffffffffffffffffff))
            revert(0, 0x24)
        }
    }

    /// @dev Reverts with a custom error with two int24 arguments
    function revertWith(bytes4 selector, int24 value1, int24 value2) internal pure {
        assembly ("memory-safe") {
            let fmp := mload(0x40)
            mstore(fmp, selector)
            mstore(add(fmp, 0x04), signextend(2, value1))
            mstore(add(fmp, 0x24), signextend(2, value2))
            revert(fmp, 0x44)
        }
    }

    /// @dev Reverts with a custom error with two uint160 arguments
    function revertWith(bytes4 selector, uint160 value1, uint160 value2) internal pure {
        assembly ("memory-safe") {
            let fmp := mload(0x40)
            mstore(fmp, selector)
            mstore(add(fmp, 0x04), and(value1, 0xffffffffffffffffffffffffffffffffffffffff))
            mstore(add(fmp, 0x24), and(value2, 0xffffffffffffffffffffffffffffffffffffffff))
            revert(fmp, 0x44)
        }
    }

    /// @dev Reverts with a custom error with two address arguments
    function revertWith(bytes4 selector, address value1, address value2) internal pure {
        assembly ("memory-safe") {
            let fmp := mload(0x40)
            mstore(fmp, selector)
            mstore(add(fmp, 0x04), and(value1, 0xffffffffffffffffffffffffffffffffffffffff))
            mstore(add(fmp, 0x24), and(value2, 0xffffffffffffffffffffffffffffffffffffffff))
            revert(fmp, 0x44)
        }
    }

    /// @notice bubble up the revert message returned by a call and revert with a wrapped ERC-7751 error
    /// @dev this method can be vulnerable to revert data bombs
    function bubbleUpAndRevertWith(
        address revertingContract,
        bytes4 revertingFunctionSelector,
        bytes4 additionalContext
    ) internal pure {
        bytes4 wrappedErrorSelector = WrappedError.selector;
        assembly ("memory-safe") {
            // Ensure the size of the revert data is a multiple of 32 bytes
            let encodedDataSize := mul(div(add(returndatasize(), 31), 32), 32)

            let fmp := mload(0x40)

            // Encode wrapped error selector, address, function selector, offset, additional context, size, revert reason
            mstore(fmp, wrappedErrorSelector)
            mstore(add(fmp, 0x04), and(revertingContract, 0xffffffffffffffffffffffffffffffffffffffff))
            mstore(
                add(fmp, 0x24),
                and(revertingFunctionSelector, 0xffffffff00000000000000000000000000000000000000000000000000000000)
            )
            // offset revert reason
            mstore(add(fmp, 0x44), 0x80)
            // offset additional context
            mstore(add(fmp, 0x64), add(0xa0, encodedDataSize))
            // size revert reason
            mstore(add(fmp, 0x84), returndatasize())
            // revert reason
            returndatacopy(add(fmp, 0xa4), 0, returndatasize())
            // size additional context
            mstore(add(fmp, add(0xa4, encodedDataSize)), 0x04)
            // additional context
            mstore(
                add(fmp, add(0xc4, encodedDataSize)),
                and(additionalContext, 0xffffffff00000000000000000000000000000000000000000000000000000000)
            )
            revert(fmp, add(0xe4, encodedDataSize))
        }
    }
}


// ===== FILE: _uniswap/v4-core/src/libraries/SafeCast.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {CustomRevert} from "./CustomRevert.sol";

/// @title Safe casting methods
/// @notice Contains methods for safely casting between types
library SafeCast {
    using CustomRevert for bytes4;

    error SafeCastOverflow();

    /// @notice Cast a uint256 to a uint160, revert on overflow
    /// @param x The uint256 to be downcasted
    /// @return y The downcasted integer, now type uint160
    function toUint160(uint256 x) internal pure returns (uint160 y) {
        y = uint160(x);
        if (y != x) SafeCastOverflow.selector.revertWith();
    }

    /// @notice Cast a uint256 to a uint128, revert on overflow
    /// @param x The uint256 to be downcasted
    /// @return y The downcasted integer, now type uint128
    function toUint128(uint256 x) internal pure returns (uint128 y) {
        y = uint128(x);
        if (x != y) SafeCastOverflow.selector.revertWith();
    }

    /// @notice Cast a int128 to a uint128, revert on overflow or underflow
    /// @param x The int128 to be casted
    /// @return y The casted integer, now type uint128
    function toUint128(int128 x) internal pure returns (uint128 y) {
        if (x < 0) SafeCastOverflow.selector.revertWith();
        y = uint128(x);
    }

    /// @notice Cast a int256 to a int128, revert on overflow or underflow
    /// @param x The int256 to be downcasted
    /// @return y The downcasted integer, now type int128
    function toInt128(int256 x) internal pure returns (int128 y) {
        y = int128(x);
        if (y != x) SafeCastOverflow.selector.revertWith();
    }

    /// @notice Cast a uint256 to a int256, revert on overflow
    /// @param x The uint256 to be casted
    /// @return y The casted integer, now type int256
    function toInt256(uint256 x) internal pure returns (int256 y) {
        y = int256(x);
        if (y < 0) SafeCastOverflow.selector.revertWith();
    }

    /// @notice Cast a uint256 to a int128, revert on overflow
    /// @param x The uint256 to be downcasted
    /// @return The downcasted integer, now type int128
    function toInt128(uint256 x) internal pure returns (int128) {
        if (x >= 1 << 127) SafeCastOverflow.selector.revertWith();
        return int128(int256(x));
    }
}


// ===== FILE: _uniswap/v4-core/src/libraries/TickMath.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {BitMath} from "./BitMath.sol";
import {CustomRevert} from "./CustomRevert.sol";

/// @title Math library for computing sqrt prices from ticks and vice versa
/// @notice Computes sqrt price for ticks of size 1.0001, i.e. sqrt(1.0001^tick) as fixed point Q64.96 numbers. Supports
/// prices between 2**-128 and 2**128
library TickMath {
    using CustomRevert for bytes4;

    /// @notice Thrown when the tick passed to #getSqrtPriceAtTick is not between MIN_TICK and MAX_TICK
    error InvalidTick(int24 tick);
    /// @notice Thrown when the price passed to #getTickAtSqrtPrice does not correspond to a price between MIN_TICK and MAX_TICK
    error InvalidSqrtPrice(uint160 sqrtPriceX96);

    /// @dev The minimum tick that may be passed to #getSqrtPriceAtTick computed from log base 1.0001 of 2**-128
    /// @dev If ever MIN_TICK and MAX_TICK are not centered around 0, the absTick logic in getSqrtPriceAtTick cannot be used
    int24 internal constant MIN_TICK = -887272;
    /// @dev The maximum tick that may be passed to #getSqrtPriceAtTick computed from log base 1.0001 of 2**128
    /// @dev If ever MIN_TICK and MAX_TICK are not centered around 0, the absTick logic in getSqrtPriceAtTick cannot be used
    int24 internal constant MAX_TICK = 887272;

    /// @dev The minimum tick spacing value drawn from the range of type int16 that is greater than 0, i.e. min from the range [1, 32767]
    int24 internal constant MIN_TICK_SPACING = 1;
    /// @dev The maximum tick spacing value drawn from the range of type int16, i.e. max from the range [1, 32767]
    int24 internal constant MAX_TICK_SPACING = type(int16).max;

    /// @dev The minimum value that can be returned from #getSqrtPriceAtTick. Equivalent to getSqrtPriceAtTick(MIN_TICK)
    uint160 internal constant MIN_SQRT_PRICE = 4295128739;
    /// @dev The maximum value that can be returned from #getSqrtPriceAtTick. Equivalent to getSqrtPriceAtTick(MAX_TICK)
    uint160 internal constant MAX_SQRT_PRICE = 1461446703485210103287273052203988822378723970342;
    /// @dev A threshold used for optimized bounds check, equals `MAX_SQRT_PRICE - MIN_SQRT_PRICE - 1`
    uint160 internal constant MAX_SQRT_PRICE_MINUS_MIN_SQRT_PRICE_MINUS_ONE =
        1461446703485210103287273052203988822378723970342 - 4295128739 - 1;

    /// @notice Given a tickSpacing, compute the maximum usable tick
    function maxUsableTick(int24 tickSpacing) internal pure returns (int24) {
        unchecked {
            return (MAX_TICK / tickSpacing) * tickSpacing;
        }
    }

    /// @notice Given a tickSpacing, compute the minimum usable tick
    function minUsableTick(int24 tickSpacing) internal pure returns (int24) {
        unchecked {
            return (MIN_TICK / tickSpacing) * tickSpacing;
        }
    }

    /// @notice Calculates sqrt(1.0001^tick) * 2^96
    /// @dev Throws if |tick| > max tick
    /// @param tick The input tick for the above formula
    /// @return sqrtPriceX96 A Fixed point Q64.96 number representing the sqrt of the price of the two assets (currency1/currency0)
    /// at the given tick
    function getSqrtPriceAtTick(int24 tick) internal pure returns (uint160 sqrtPriceX96) {
        unchecked {
            uint256 absTick;
            assembly ("memory-safe") {
                tick := signextend(2, tick)
                // mask = 0 if tick >= 0 else -1 (all 1s)
                let mask := sar(255, tick)
                // if tick >= 0, |tick| = tick = 0 ^ tick
                // if tick < 0, |tick| = ~~|tick| = ~(-|tick| - 1) = ~(tick - 1) = (-1) ^ (tick - 1)
                // either way, |tick| = mask ^ (tick + mask)
                absTick := xor(mask, add(mask, tick))
            }

            if (absTick > uint256(int256(MAX_TICK))) InvalidTick.selector.revertWith(tick);

            // The tick is decomposed into bits, and for each bit with index i that is set, the product of 1/sqrt(1.0001^(2^i))
            // is calculated (using Q128.128). The constants used for this calculation are rounded to the nearest integer

            // Equivalent to:
            //     price = absTick & 0x1 != 0 ? 0xfffcb933bd6fad37aa2d162d1a594001 : 0x100000000000000000000000000000000;
            //     or price = int(2**128 / sqrt(1.0001)) if (absTick & 0x1) else 1 << 128
            uint256 price;
            assembly ("memory-safe") {
                price := xor(shl(128, 1), mul(xor(shl(128, 1), 0xfffcb933bd6fad37aa2d162d1a594001), and(absTick, 0x1)))
            }
            if (absTick & 0x2 != 0) price = (price * 0xfff97272373d413259a46990580e213a) >> 128;
            if (absTick & 0x4 != 0) price = (price * 0xfff2e50f5f656932ef12357cf3c7fdcc) >> 128;
            if (absTick & 0x8 != 0) price = (price * 0xffe5caca7e10e4e61c3624eaa0941cd0) >> 128;
            if (absTick & 0x10 != 0) price = (price * 0xffcb9843d60f6159c9db58835c926644) >> 128;
            if (absTick & 0x20 != 0) price = (price * 0xff973b41fa98c081472e6896dfb254c0) >> 128;
            if (absTick & 0x40 != 0) price = (price * 0xff2ea16466c96a3843ec78b326b52861) >> 128;
            if (absTick & 0x80 != 0) price = (price * 0xfe5dee046a99a2a811c461f1969c3053) >> 128;
            if (absTick & 0x100 != 0) price = (price * 0xfcbe86c7900a88aedcffc83b479aa3a4) >> 128;
            if (absTick & 0x200 != 0) price = (price * 0xf987a7253ac413176f2b074cf7815e54) >> 128;
            if (absTick & 0x400 != 0) price = (price * 0xf3392b0822b70005940c7a398e4b70f3) >> 128;
            if (absTick & 0x800 != 0) price = (price * 0xe7159475a2c29b7443b29c7fa6e889d9) >> 128;
            if (absTick & 0x1000 != 0) price = (price * 0xd097f3bdfd2022b8845ad8f792aa5825) >> 128;
            if (absTick & 0x2000 != 0) price = (price * 0xa9f746462d870fdf8a65dc1f90e061e5) >> 128;
            if (absTick & 0x4000 != 0) price = (price * 0x70d869a156d2a1b890bb3df62baf32f7) >> 128;
            if (absTick & 0x8000 != 0) price = (price * 0x31be135f97d08fd981231505542fcfa6) >> 128;
            if (absTick & 0x10000 != 0) price = (price * 0x9aa508b5b7a84e1c677de54f3e99bc9) >> 128;
            if (absTick & 0x20000 != 0) price = (price * 0x5d6af8dedb81196699c329225ee604) >> 128;
            if (absTick & 0x40000 != 0) price = (price * 0x2216e584f5fa1ea926041bedfe98) >> 128;
            if (absTick & 0x80000 != 0) price = (price * 0x48a170391f7dc42444e8fa2) >> 128;

            assembly ("memory-safe") {
                // if (tick > 0) price = type(uint256).max / price;
                if sgt(tick, 0) { price := div(not(0), price) }

                // this divides by 1<<32 rounding up to go from a Q128.128 to a Q128.96.
                // we then downcast because we know the result always fits within 160 bits due to our tick input constraint
                // we round up in the division so getTickAtSqrtPrice of the output price is always consistent
                // `sub(shl(32, 1), 1)` is `type(uint32).max`
                // `price + type(uint32).max` will not overflow because `price` fits in 192 bits
                sqrtPriceX96 := shr(32, add(price, sub(shl(32, 1), 1)))
            }
        }
    }

    /// @notice Calculates the greatest tick value such that getSqrtPriceAtTick(tick) <= sqrtPriceX96
    /// @dev Throws in case sqrtPriceX96 < MIN_SQRT_PRICE, as MIN_SQRT_PRICE is the lowest value getSqrtPriceAtTick may
    /// ever return.
    /// @param sqrtPriceX96 The sqrt price for which to compute the tick as a Q64.96
    /// @return tick The greatest tick for which the getSqrtPriceAtTick(tick) is less than or equal to the input sqrtPriceX96
    function getTickAtSqrtPrice(uint160 sqrtPriceX96) internal pure returns (int24 tick) {
        unchecked {
            // Equivalent: if (sqrtPriceX96 < MIN_SQRT_PRICE || sqrtPriceX96 >= MAX_SQRT_PRICE) revert InvalidSqrtPrice();
            // second inequality must be >= because the price can never reach the price at the max tick
            // if sqrtPriceX96 < MIN_SQRT_PRICE, the `sub` underflows and `gt` is true
            // if sqrtPriceX96 >= MAX_SQRT_PRICE, sqrtPriceX96 - MIN_SQRT_PRICE > MAX_SQRT_PRICE - MIN_SQRT_PRICE - 1
            if ((sqrtPriceX96 - MIN_SQRT_PRICE) > MAX_SQRT_PRICE_MINUS_MIN_SQRT_PRICE_MINUS_ONE) {
                InvalidSqrtPrice.selector.revertWith(sqrtPriceX96);
            }

            uint256 price = uint256(sqrtPriceX96) << 32;

            uint256 r = price;
            uint256 msb = BitMath.mostSignificantBit(r);

            if (msb >= 128) r = price >> (msb - 127);
            else r = price << (127 - msb);

            int256 log_2 = (int256(msb) - 128) << 64;

            assembly ("memory-safe") {
                r := shr(127, mul(r, r))
                let f := shr(128, r)
                log_2 := or(log_2, shl(63, f))
                r := shr(f, r)
            }
            assembly ("memory-safe") {
                r := shr(127, mul(r, r))
                let f := shr(128, r)
                log_2 := or(log_2, shl(62, f))
                r := shr(f, r)
            }
            assembly ("memory-safe") {
                r := shr(127, mul(r, r))
                let f := shr(128, r)
                log_2 := or(log_2, shl(61, f))
                r := shr(f, r)
            }
            assembly ("memory-safe") {
                r := shr(127, mul(r, r))
                let f := shr(128, r)
                log_2 := or(log_2, shl(60, f))
                r := shr(f, r)
            }
            assembly ("memory-safe") {
                r := shr(127, mul(r, r))
                let f := shr(128, r)
                log_2 := or(log_2, shl(59, f))
                r := shr(f, r)
            }
            assembly ("memory-safe") {
                r := shr(127, mul(r, r))
                let f := shr(128, r)
                log_2 := or(log_2, shl(58, f))
                r := shr(f, r)
            }
            assembly ("memory-safe") {
                r := shr(127, mul(r, r))
                let f := shr(128, r)
                log_2 := or(log_2, shl(57, f))
                r := shr(f, r)
            }
            assembly ("memory-safe") {
                r := shr(127, mul(r, r))
                let f := shr(128, r)
                log_2 := or(log_2, shl(56, f))
                r := shr(f, r)
            }
            assembly ("memory-safe") {
                r := shr(127, mul(r, r))
                let f := shr(128, r)
                log_2 := or(log_2, shl(55, f))
                r := shr(f, r)
            }
            assembly ("memory-safe") {
                r := shr(127, mul(r, r))
                let f := shr(128, r)
                log_2 := or(log_2, shl(54, f))
                r := shr(f, r)
            }
            assembly ("memory-safe") {
                r := shr(127, mul(r, r))
                let f := shr(128, r)
                log_2 := or(log_2, shl(53, f))
                r := shr(f, r)
            }
            assembly ("memory-safe") {
                r := shr(127, mul(r, r))
                let f := shr(128, r)
                log_2 := or(log_2, shl(52, f))
                r := shr(f, r)
            }
            assembly ("memory-safe") {
                r := shr(127, mul(r, r))
                let f := shr(128, r)
                log_2 := or(log_2, shl(51, f))
                r := shr(f, r)
            }
            assembly ("memory-safe") {
                r := shr(127, mul(r, r))
                let f := shr(128, r)
                log_2 := or(log_2, shl(50, f))
            }

            int256 log_sqrt10001 = log_2 * 255738958999603826347141; // Q22.128 number

            // Magic number represents the ceiling of the maximum value of the error when approximating log_sqrt10001(x)
            int24 tickLow = int24((log_sqrt10001 - 3402992956809132418596140100660247210) >> 128);

            // Magic number represents the minimum value of the error when approximating log_sqrt10001(x), when
            // sqrtPrice is from the range (2^-64, 2^64). This is safe as MIN_SQRT_PRICE is more than 2^-64. If MIN_SQRT_PRICE
            // is changed, this may need to be changed too
            int24 tickHi = int24((log_sqrt10001 + 291339464771989622907027621153398088495) >> 128);

            tick = tickLow == tickHi ? tickLow : getSqrtPriceAtTick(tickHi) <= sqrtPriceX96 ? tickHi : tickLow;
        }
    }
}


// ===== FILE: _uniswap/v4-core/src/types/BalanceDelta.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {SafeCast} from "../libraries/SafeCast.sol";

/// @dev Two `int128` values packed into a single `int256` where the upper 128 bits represent the amount0
/// and the lower 128 bits represent the amount1.
type BalanceDelta is int256;

using {add as +, sub as -, eq as ==, neq as !=} for BalanceDelta global;
using BalanceDeltaLibrary for BalanceDelta global;
using SafeCast for int256;

function toBalanceDelta(int128 _amount0, int128 _amount1) pure returns (BalanceDelta balanceDelta) {
    assembly ("memory-safe") {
        balanceDelta := or(shl(128, _amount0), and(sub(shl(128, 1), 1), _amount1))
    }
}

function add(BalanceDelta a, BalanceDelta b) pure returns (BalanceDelta) {
    int256 res0;
    int256 res1;
    assembly ("memory-safe") {
        let a0 := sar(128, a)
        let a1 := signextend(15, a)
        let b0 := sar(128, b)
        let b1 := signextend(15, b)
        res0 := add(a0, b0)
        res1 := add(a1, b1)
    }
    return toBalanceDelta(res0.toInt128(), res1.toInt128());
}

function sub(BalanceDelta a, BalanceDelta b) pure returns (BalanceDelta) {
    int256 res0;
    int256 res1;
    assembly ("memory-safe") {
        let a0 := sar(128, a)
        let a1 := signextend(15, a)
        let b0 := sar(128, b)
        let b1 := signextend(15, b)
        res0 := sub(a0, b0)
        res1 := sub(a1, b1)
    }
    return toBalanceDelta(res0.toInt128(), res1.toInt128());
}

function eq(BalanceDelta a, BalanceDelta b) pure returns (bool) {
    return BalanceDelta.unwrap(a) == BalanceDelta.unwrap(b);
}

function neq(BalanceDelta a, BalanceDelta b) pure returns (bool) {
    return BalanceDelta.unwrap(a) != BalanceDelta.unwrap(b);
}

/// @notice Library for getting the amount0 and amount1 deltas from the BalanceDelta type
library BalanceDeltaLibrary {
    /// @notice A BalanceDelta of 0
    BalanceDelta public constant ZERO_DELTA = BalanceDelta.wrap(0);

    function amount0(BalanceDelta balanceDelta) internal pure returns (int128 _amount0) {
        assembly ("memory-safe") {
            _amount0 := sar(128, balanceDelta)
        }
    }

    function amount1(BalanceDelta balanceDelta) internal pure returns (int128 _amount1) {
        assembly ("memory-safe") {
            _amount1 := signextend(15, balanceDelta)
        }
    }
}


// ===== FILE: _uniswap/v4-core/src/types/BeforeSwapDelta.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Return type of the beforeSwap hook.
// Upper 128 bits is the delta in specified tokens. Lower 128 bits is delta in unspecified tokens (to match the afterSwap hook)
type BeforeSwapDelta is int256;

// Creates a BeforeSwapDelta from specified and unspecified
function toBeforeSwapDelta(int128 deltaSpecified, int128 deltaUnspecified)
    pure
    returns (BeforeSwapDelta beforeSwapDelta)
{
    assembly ("memory-safe") {
        beforeSwapDelta := or(shl(128, deltaSpecified), and(sub(shl(128, 1), 1), deltaUnspecified))
    }
}

/// @notice Library for getting the specified and unspecified deltas from the BeforeSwapDelta type
library BeforeSwapDeltaLibrary {
    /// @notice A BeforeSwapDelta of 0
    BeforeSwapDelta public constant ZERO_DELTA = BeforeSwapDelta.wrap(0);

    /// extracts int128 from the upper 128 bits of the BeforeSwapDelta
    /// returned by beforeSwap
    function getSpecifiedDelta(BeforeSwapDelta delta) internal pure returns (int128 deltaSpecified) {
        assembly ("memory-safe") {
            deltaSpecified := sar(128, delta)
        }
    }

    /// extracts int128 from the lower 128 bits of the BeforeSwapDelta
    /// returned by beforeSwap and afterSwap
    function getUnspecifiedDelta(BeforeSwapDelta delta) internal pure returns (int128 deltaUnspecified) {
        assembly ("memory-safe") {
            deltaUnspecified := signextend(15, delta)
        }
    }
}


// ===== FILE: _uniswap/v4-core/src/types/Currency.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {IERC20Minimal} from "../interfaces/external/IERC20Minimal.sol";
import {CustomRevert} from "../libraries/CustomRevert.sol";

type Currency is address;

using {greaterThan as >, lessThan as <, greaterThanOrEqualTo as >=, equals as ==} for Currency global;
using CurrencyLibrary for Currency global;

function equals(Currency currency, Currency other) pure returns (bool) {
    return Currency.unwrap(currency) == Currency.unwrap(other);
}

function greaterThan(Currency currency, Currency other) pure returns (bool) {
    return Currency.unwrap(currency) > Currency.unwrap(other);
}

function lessThan(Currency currency, Currency other) pure returns (bool) {
    return Currency.unwrap(currency) < Currency.unwrap(other);
}

function greaterThanOrEqualTo(Currency currency, Currency other) pure returns (bool) {
    return Currency.unwrap(currency) >= Currency.unwrap(other);
}

/// @title CurrencyLibrary
/// @dev This library allows for transferring and holding native tokens and ERC20 tokens
library CurrencyLibrary {
    /// @notice Additional context for ERC-7751 wrapped error when a native transfer fails
    error NativeTransferFailed();

    /// @notice Additional context for ERC-7751 wrapped error when an ERC20 transfer fails
    error ERC20TransferFailed();

    /// @notice A constant to represent the native currency
    Currency public constant ADDRESS_ZERO = Currency.wrap(address(0));

    function transfer(Currency currency, address to, uint256 amount) internal {
        // altered from https://github.com/transmissions11/solmate/blob/44a9963d4c78111f77caa0e65d677b8b46d6f2e6/src/utils/SafeTransferLib.sol
        // modified custom error selectors

        bool success;
        if (currency.isAddressZero()) {
            assembly ("memory-safe") {
                // Transfer the ETH and revert if it fails.
                success := call(gas(), to, amount, 0, 0, 0, 0)
            }
            // revert with NativeTransferFailed, containing the bubbled up error as an argument
            if (!success) {
                CustomRevert.bubbleUpAndRevertWith(to, bytes4(0), NativeTransferFailed.selector);
            }
        } else {
            assembly ("memory-safe") {
                // Get a pointer to some free memory.
                let fmp := mload(0x40)

                // Write the abi-encoded calldata into memory, beginning with the function selector.
                mstore(fmp, 0xa9059cbb00000000000000000000000000000000000000000000000000000000)
                mstore(add(fmp, 4), and(to, 0xffffffffffffffffffffffffffffffffffffffff)) // Append and mask the "to" argument.
                mstore(add(fmp, 36), amount) // Append the "amount" argument. Masking not required as it's a full 32 byte type.

                success :=
                    and(
                        // Set success to whether the call reverted, if not we check it either
                        // returned exactly 1 (can't just be non-zero data), or had no return data.
                        or(and(eq(mload(0), 1), gt(returndatasize(), 31)), iszero(returndatasize())),
                        // We use 68 because the length of our calldata totals up like so: 4 + 32 * 2.
                        // We use 0 and 32 to copy up to 32 bytes of return data into the scratch space.
                        // Counterintuitively, this call must be positioned second to the or() call in the
                        // surrounding and() call or else returndatasize() will be zero during the computation.
                        call(gas(), currency, 0, fmp, 68, 0, 32)
                    )

                // Now clean the memory we used
                mstore(fmp, 0) // 4 byte `selector` and 28 bytes of `to` were stored here
                mstore(add(fmp, 0x20), 0) // 4 bytes of `to` and 28 bytes of `amount` were stored here
                mstore(add(fmp, 0x40), 0) // 4 bytes of `amount` were stored here
            }
            // revert with ERC20TransferFailed, containing the bubbled up error as an argument
            if (!success) {
                CustomRevert.bubbleUpAndRevertWith(
                    Currency.unwrap(currency), IERC20Minimal.transfer.selector, ERC20TransferFailed.selector
                );
            }
        }
    }

    function balanceOfSelf(Currency currency) internal view returns (uint256) {
        if (currency.isAddressZero()) {
            return address(this).balance;
        } else {
            return IERC20Minimal(Currency.unwrap(currency)).balanceOf(address(this));
        }
    }

    function balanceOf(Currency currency, address owner) internal view returns (uint256) {
        if (currency.isAddressZero()) {
            return owner.balance;
        } else {
            return IERC20Minimal(Currency.unwrap(currency)).balanceOf(owner);
        }
    }

    function isAddressZero(Currency currency) internal pure returns (bool) {
        return Currency.unwrap(currency) == Currency.unwrap(ADDRESS_ZERO);
    }

    function toId(Currency currency) internal pure returns (uint256) {
        return uint160(Currency.unwrap(currency));
    }

    // If the upper 12 bytes are non-zero, they will be zero-ed out
    // Therefore, fromId() and toId() are not inverses of each other
    function fromId(uint256 id) internal pure returns (Currency) {
        return Currency.wrap(address(uint160(id)));
    }
}


// ===== FILE: _uniswap/v4-core/src/types/PoolId.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {PoolKey} from "./PoolKey.sol";

type PoolId is bytes32;

/// @notice Library for computing the ID of a pool
library PoolIdLibrary {
    /// @notice Returns value equal to keccak256(abi.encode(poolKey))
    function toId(PoolKey memory poolKey) internal pure returns (PoolId poolId) {
        assembly ("memory-safe") {
            // 0xa0 represents the total size of the poolKey struct (5 slots of 32 bytes)
            poolId := keccak256(poolKey, 0xa0)
        }
    }
}


// ===== FILE: _uniswap/v4-core/src/types/PoolKey.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {Currency} from "./Currency.sol";
import {IHooks} from "../interfaces/IHooks.sol";
import {PoolIdLibrary} from "./PoolId.sol";

using PoolIdLibrary for PoolKey global;

/// @notice Returns the key for identifying a pool
struct PoolKey {
    /// @notice The lower currency of the pool, sorted numerically
    Currency currency0;
    /// @notice The higher currency of the pool, sorted numerically
    Currency currency1;
    /// @notice The pool LP fee, capped at 1_000_000. If the highest bit is 1, the pool has a dynamic fee and must be exactly equal to 0x800000
    uint24 fee;
    /// @notice Ticks that involve positions must be a multiple of tick spacing
    int24 tickSpacing;
    /// @notice The hooks of the pool
    IHooks hooks;
}


// ===== FILE: _uniswap/v4-core/src/types/PoolOperation.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {PoolKey} from "../types/PoolKey.sol";
import {BalanceDelta} from "../types/BalanceDelta.sol";

/// @notice Parameter struct for `ModifyLiquidity` pool operations
struct ModifyLiquidityParams {
    // the lower and upper tick of the position
    int24 tickLower;
    int24 tickUpper;
    // how to modify the liquidity
    int256 liquidityDelta;
    // a value to set if you want unique liquidity positions at the same range
    bytes32 salt;
}

/// @notice Parameter struct for `Swap` pool operations
struct SwapParams {
    /// Whether to swap token0 for token1 or vice versa
    bool zeroForOne;
    /// The desired input amount if negative (exactIn), or the desired output amount if positive (exactOut)
    int256 amountSpecified;
    /// The sqrt price at which, if reached, the swap will stop executing
    uint160 sqrtPriceLimitX96;
}


// ===== FILE: _uniswap/v4-periphery/src/interfaces/IPoolInitializer_v4.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {PoolKey} from "@uniswap/v4-core/src/types/PoolKey.sol";

/// @title IPoolInitializer_v4
/// @notice Interface for the PoolInitializer_v4 contract
interface IPoolInitializer_v4 {
    /// @notice Initialize a Uniswap v4 Pool
    /// @dev If the pool is already initialized, this function will not revert and just return type(int24).max
    /// @param key The PoolKey of the pool to initialize
    /// @param sqrtPriceX96 The initial starting price of the pool, expressed as a sqrtPriceX96
    /// @return The current tick of the pool, or type(int24).max if the pool creation failed, or the pool already existed
    function initializePool(PoolKey calldata key, uint160 sqrtPriceX96) external payable returns (int24);
}


// ===== FILE: _uniswap/v4-periphery/src/libraries/Actions.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @notice Library to define different pool actions.
/// @dev These are suggested common commands, however additional commands should be defined as required
/// Some of these actions are not supported in the Router contracts or Position Manager contracts, but are left as they may be helpful commands for other peripheral contracts.
library Actions {
    // pool actions
    // liquidity actions
    uint256 internal constant INCREASE_LIQUIDITY = 0x00;
    uint256 internal constant DECREASE_LIQUIDITY = 0x01;
    uint256 internal constant MINT_POSITION = 0x02;
    uint256 internal constant BURN_POSITION = 0x03;
    uint256 internal constant INCREASE_LIQUIDITY_FROM_DELTAS = 0x04;
    uint256 internal constant MINT_POSITION_FROM_DELTAS = 0x05;

    // swapping
    uint256 internal constant SWAP_EXACT_IN_SINGLE = 0x06;
    uint256 internal constant SWAP_EXACT_IN = 0x07;
    uint256 internal constant SWAP_EXACT_OUT_SINGLE = 0x08;
    uint256 internal constant SWAP_EXACT_OUT = 0x09;

    // donate
    // note this is not supported in the position manager or router
    uint256 internal constant DONATE = 0x0a;

    // closing deltas on the pool manager
    // settling
    uint256 internal constant SETTLE = 0x0b;
    uint256 internal constant SETTLE_ALL = 0x0c;
    uint256 internal constant SETTLE_PAIR = 0x0d;
    // taking
    uint256 internal constant TAKE = 0x0e;
    uint256 internal constant TAKE_ALL = 0x0f;
    uint256 internal constant TAKE_PORTION = 0x10;
    uint256 internal constant TAKE_PAIR = 0x11;

    uint256 internal constant CLOSE_CURRENCY = 0x12;
    uint256 internal constant CLEAR_OR_TAKE = 0x13;
    uint256 internal constant SWEEP = 0x14;

    uint256 internal constant WRAP = 0x15;
    uint256 internal constant UNWRAP = 0x16;

    // minting/burning 6909s to close deltas
    // note this is not supported in the position manager or router
    uint256 internal constant MINT_6909 = 0x17;
    uint256 internal constant BURN_6909 = 0x18;
}


// ===== FILE: _uniswap/v4-periphery/src/libraries/PositionInfoLibrary.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {PoolKey} from "@uniswap/v4-core/src/types/PoolKey.sol";
import {PoolId} from "@uniswap/v4-core/src/types/PoolId.sol";

/**
 * @dev PositionInfo is a packed version of solidity structure.
 * Using the packaged version saves gas and memory by not storing the structure fields in memory slots.
 *
 * Layout:
 * 200 bits poolId | 24 bits tickUpper | 24 bits tickLower | 8 bits hasSubscriber
 *
 * Fields in the direction from the least significant bit:
 *
 * A flag to know if the tokenId is subscribed to an address
 * uint8 hasSubscriber;
 *
 * The tickUpper of the position
 * int24 tickUpper;
 *
 * The tickLower of the position
 * int24 tickLower;
 *
 * The truncated poolId. Truncates a bytes32 value so the most signifcant (highest) 200 bits are used.
 * bytes25 poolId;
 *
 * Note: If more bits are needed, hasSubscriber can be a single bit.
 *
 */
type PositionInfo is uint256;

using PositionInfoLibrary for PositionInfo global;

library PositionInfoLibrary {
    PositionInfo internal constant EMPTY_POSITION_INFO = PositionInfo.wrap(0);

    uint256 internal constant MASK_UPPER_200_BITS = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000;
    uint256 internal constant MASK_8_BITS = 0xFF;
    uint24 internal constant MASK_24_BITS = 0xFFFFFF;
    uint256 internal constant SET_UNSUBSCRIBE = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00;
    uint256 internal constant SET_SUBSCRIBE = 0x01;
    uint8 internal constant TICK_LOWER_OFFSET = 8;
    uint8 internal constant TICK_UPPER_OFFSET = 32;

    /// @dev This poolId is NOT compatible with the poolId used in UniswapV4 core. It is truncated to 25 bytes, and just used to lookup PoolKey in the poolKeys mapping.
    function poolId(PositionInfo info) internal pure returns (bytes25 _poolId) {
        assembly ("memory-safe") {
            _poolId := and(MASK_UPPER_200_BITS, info)
        }
    }

    function tickLower(PositionInfo info) internal pure returns (int24 _tickLower) {
        assembly ("memory-safe") {
            _tickLower := signextend(2, shr(TICK_LOWER_OFFSET, info))
        }
    }

    function tickUpper(PositionInfo info) internal pure returns (int24 _tickUpper) {
        assembly ("memory-safe") {
            _tickUpper := signextend(2, shr(TICK_UPPER_OFFSET, info))
        }
    }

    function hasSubscriber(PositionInfo info) internal pure returns (bool _hasSubscriber) {
        assembly ("memory-safe") {
            _hasSubscriber := and(MASK_8_BITS, info)
        }
    }

    /// @dev this does not actually set any storage
    function setSubscribe(PositionInfo info) internal pure returns (PositionInfo _info) {
        assembly ("memory-safe") {
            _info := or(info, SET_SUBSCRIBE)
        }
    }

    /// @dev this does not actually set any storage
    function setUnsubscribe(PositionInfo info) internal pure returns (PositionInfo _info) {
        assembly ("memory-safe") {
            _info := and(info, SET_UNSUBSCRIBE)
        }
    }

    /// @notice Creates the default PositionInfo struct
    /// @dev Called when minting a new position
    /// @param _poolKey the pool key of the position
    /// @param _tickLower the lower tick of the position
    /// @param _tickUpper the upper tick of the position
    /// @return info packed position info, with the truncated poolId and the hasSubscriber flag set to false
    function initialize(PoolKey memory _poolKey, int24 _tickLower, int24 _tickUpper)
        internal
        pure
        returns (PositionInfo info)
    {
        bytes25 _poolId = bytes25(PoolId.unwrap(_poolKey.toId()));
        assembly {
            info :=
                or(
                    or(and(MASK_UPPER_200_BITS, _poolId), shl(TICK_UPPER_OFFSET, and(MASK_24_BITS, _tickUpper))),
                    shl(TICK_LOWER_OFFSET, and(MASK_24_BITS, _tickLower))
                )
        }
    }
}


// ===== FILE: contracts/main/interfaces/IPoolManagerMinimal.sol =====
// SPDX-License-Identifier: MIT
//  Minimal interface for Uniswap V4 PoolManager interactions.
pragma solidity ^0.8.26;

import {PoolKey}      from "@uniswap/v4-core/src/types/PoolKey.sol";
import {PoolId}       from "@uniswap/v4-core/src/types/PoolId.sol";
import {Currency}     from "@uniswap/v4-core/src/types/Currency.sol";
import {BalanceDelta} from "@uniswap/v4-core/src/types/BalanceDelta.sol";

/**
 * @notice Minimal subset of Uniswap V4 PoolManager needed by ZephyrFactory.
 *         Defines SwapParams inline to avoid importing PoolOperation.sol,
 *         which some environments (e.g. Remix) cannot resolve via npm path.
 */
interface IPoolManagerMinimal {
    struct SwapParams {
        bool    zeroForOne;
        int256  amountSpecified;
        uint160 sqrtPriceLimitX96;
    }

    function unlock(bytes calldata data) external returns (bytes memory);

    function swap(
        PoolKey memory key,
        SwapParams memory params,
        bytes calldata hookData
    ) external returns (BalanceDelta);

    function sync(Currency currency) external;

    function settle() external payable returns (uint256);

    function take(Currency currency, address to, uint256 amount) external;

    function getSlot0(PoolId id) external view returns (
        uint160 sqrtPriceX96,
        int24   tick,
        uint24  protocolFee,
        uint24  lpFee
    );

    function getLiquidity(PoolId id) external view returns (uint128 liquidity);
}


// ===== FILE: contracts/main/interfaces/IPositionManagerMinimal.sol =====
// SPDX-License-Identifier: MIT
//  Minimal interface for Uniswap V4 PositionManager interactions.
pragma solidity ^0.8.26;

import {PoolKey} from "@uniswap/v4-core/src/types/PoolKey.sol";
import {PositionInfo} from "@uniswap/v4-periphery/src/libraries/PositionInfoLibrary.sol";

/**
 * @notice Minimal subset of Uniswap V4 PositionManager needed by Zephyr contracts.
 *         Avoids pulling in the full IPositionManager → IPermit2Forwarder chain.
 */
interface IPositionManagerMinimal {
    /// @notice Batch multiple calls in a single transaction (payable).
    function multicall(bytes[] calldata data)
        external
        payable
        returns (bytes[] memory results);

    /// @notice Unlock the PoolManager and execute a batch of liquidity actions.
    function modifyLiquidities(bytes calldata unlockData, uint256 deadline)
        external
        payable;

    /// @notice The token ID that will be used for the next minted position.
    function nextTokenId() external view returns (uint256);

    /// @notice Returns the PoolKey and position info for a given LP NFT.
    ///         V4 equivalent of V3's positions(tokenId) — use to get currency0/currency1.
    function getPoolAndPositionInfo(uint256 tokenId)
        external
        view
        returns (PoolKey memory poolKey, PositionInfo positionInfo);

}


// ===== FILE: contracts/main/interfaces/ITokenDeployer.sol =====
// SPDX-License-Identifier: MIT
//  Interface for token deployer contracts.
pragma solidity ^0.8.26;

/**
 * @notice Common interface for token deployer contracts.
 *         Each implementation deploys a different token variant
 *         (standard, anti-sniper, etc.) and returns its address.
 */
interface ITokenDeployer {
    function deploy(
        string  calldata name,
        string  calldata symbol,
        uint256          supply,
        address          poolManager,
        address          devWallet
    ) external returns (address token);
}


// ===== FILE: contracts/main/SatFactory.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

// Deploy your own ERC-20 token with SAT Protocol.
// Built on Uniswap V4.
// SAT Protocol introduces a structured launch framework designed around transparency, security, and immutable deployment standards.
//
// https://x.com/satprotocol_io
// https://satprotocol.io/
// https://t.me/satprotocol_io

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";

import {PoolKey}             from "@uniswap/v4-core/src/types/PoolKey.sol";
import {PoolId, PoolIdLibrary} from "@uniswap/v4-core/src/types/PoolId.sol";
import {Currency, CurrencyLibrary} from "@uniswap/v4-core/src/types/Currency.sol";
import {IHooks}              from "@uniswap/v4-core/src/interfaces/IHooks.sol";
import {IUnlockCallback}     from "@uniswap/v4-core/src/interfaces/callback/IUnlockCallback.sol";
import {BalanceDelta, BalanceDeltaLibrary} from "@uniswap/v4-core/src/types/BalanceDelta.sol";
import {IPoolManagerMinimal} from "./interfaces/IPoolManagerMinimal.sol";
import {IPoolInitializer_v4} from "@uniswap/v4-periphery/src/interfaces/IPoolInitializer_v4.sol";
import {Actions}             from "@uniswap/v4-periphery/src/libraries/Actions.sol";
import {TickMath}            from "@uniswap/v4-core/src/libraries/TickMath.sol";
import {IPositionManagerMinimal} from "./interfaces/IPositionManagerMinimal.sol";

import {ITokenDeployer} from "./interfaces/ITokenDeployer.sol";
import {SatLocker}      from "./SatLocker.sol";

interface IWETH {
    function deposit() external payable;
    function transfer(address to, uint256 value) external returns (bool);
}

interface IAgentsMinter {
    function mintFromFactory(address to) external returns (uint256 tokenId);
}

/**
 * @title  SatFactory
 * @notice Single-tx project launcher: deploys an ERC20, mints a Sat Protocol Agent NFT,
 *         creates a Uniswap V4 pool seeded with a one-sided LP, locks the LP, and runs
 *         an optional dev buy.
 *
 *         Every token has a fixed 1% pool fee (FIXED_FEE = 10_000).
 *         Fee routing: platform takes 50% of collected LP fees; the rest goes to the
 *         Agent NFT owner (looked up via ownerOf at claim time).
 */
contract SatFactory is Ownable, IUnlockCallback {
    using PoolIdLibrary for PoolKey;

    // ─────────────────────────────────────────────────────────────
    //  Constants
    // ─────────────────────────────────────────────────────────────

    uint256 public constant STANDARD_SUPPLY = 1_000_000_000 * 10 ** 18;
    int24   public constant TICK_RANGE      = 110_400;
    int24   public constant TICK_SPACING    = 200;
    uint24  public constant FIXED_FEE       = 10_000; // 1%

    // ─────────────────────────────────────────────────────────────
    //  Types
    // ─────────────────────────────────────────────────────────────

    struct TokenInfo {
        address token;
        address creator;
        uint256 deployedAt;
        uint256 positionId;
        address paired;
        PoolId  poolId;
        uint24  fee;
        string  name;
        string  symbol;
        string  twitter;
        string  telegram;
        string  website;
        uint256 agentTokenId;
    }

    struct DeployParams {
        string  name;
        string  symbol;
        uint24  startingTick;
        address pairedToken;
        uint256 devBuyAmountIn;
        string  twitter;   // max 50 chars
        string  telegram;  // max 50 chars
        string  website;   // max 50 chars
    }

    struct DevBuyData {
        PoolKey  key;
        bool     zeroForOne;
        uint256  amountIn;
        Currency pairedCurrency;
        address  recipient;
    }

    // ─────────────────────────────────────────────────────────────
    //  State
    // ─────────────────────────────────────────────────────────────

    IPositionManagerMinimal public immutable positionManager;
    IPoolManagerMinimal     public immutable poolManager;
    address                 public immutable weth;

    ITokenDeployer public tokenDeployer;
    SatLocker      public locker;
    IAgentsMinter  public agents;

    TokenInfo[] private _tokens;

    mapping(address => uint256) public tokenIndex;
    mapping(uint256 => address) public tokenByAgentId;
    mapping(address => bool)    public allowedPairs;
    mapping(address => uint24[]) private _validTicks;

    uint256 public deploymentFee = 0.000001 ether;

    // ─────────────────────────────────────────────────────────────
    //  Events / Errors
    // ─────────────────────────────────────────────────────────────

    event TokenDeployed(
        address indexed token,
        address indexed creator,
        address indexed paired,
        uint256 positionId,
        PoolId  poolId,
        uint24  fee,
        string  name,
        string  symbol,
        string  twitter,
        string  telegram,
        string  website,
        uint256 agentTokenId,
        uint256 deployedAt
    );
    event LockerSet        (address indexed locker);
    event AgentsSet        (address indexed agents);
    event DeployerSet      (address deployer);
    event PairConfigured   (address indexed paired, bool allowed, uint24[] ticks);
    event DeploymentFeeSet (uint256 fee);
    event ETHWithdrawn     (address indexed to, uint256 amount);

    error TokenNotFound(address token);
    error DeployerNotSet();
    error AgentsNotSet();
    error LockerNotSet();
    error InsufficientDeploymentFee();
    error DevBuyRequiresWETHPair();
    error DevBuyAmbiguousInput();
    error PairNotAllowed(address paired);
    error InvalidStartingTick(uint24 tick, address paired);
    error StringTooLong(string field);
    error ZeroAddress();

    // ─────────────────────────────────────────────────────────────
    //  Constructor
    // ─────────────────────────────────────────────────────────────

    constructor(
        address positionManager_,
        address poolManager_,
        address weth_
    ) Ownable(msg.sender) {
        positionManager = IPositionManagerMinimal(positionManager_);
        poolManager     = IPoolManagerMinimal(poolManager_);
        weth            = weth_;
    }

    // ─────────────────────────────────────────────────────────────
    //  Admin
    // ─────────────────────────────────────────────────────────────

    function setLocker(address locker_) external onlyOwner {
        locker = SatLocker(locker_);
        emit LockerSet(locker_);
    }

    function setAgents(address agents_) external onlyOwner {
        agents = IAgentsMinter(agents_);
        emit AgentsSet(agents_);
    }

    function setDeployer(address deployer_) external onlyOwner {
        tokenDeployer = ITokenDeployer(deployer_);
        emit DeployerSet(deployer_);
    }

    function configurePair(
        address  paired,
        bool     allowed,
        uint24[] calldata ticks
    ) external onlyOwner {
        allowedPairs[paired] = allowed;
        if (ticks.length > 0) _validTicks[paired] = ticks;
        emit PairConfigured(paired, allowed, ticks);
    }

    function getValidTicks(address paired) external view returns (uint24[] memory) {
        return _validTicks[paired];
    }

    function _isValidTick(address paired, uint24 tick) internal view returns (bool) {
        uint24[] storage ticks = _validTicks[paired];
        for (uint256 i; i < ticks.length; i++) {
            if (ticks[i] == tick) return true;
        }
        return false;
    }

    // ─────────────────────────────────────────────────────────────
    //  Core: deploy
    // ─────────────────────────────────────────────────────────────

    function deploy(DeployParams calldata p)
        external
        payable
        returns (address token, uint256 positionId, uint256 tokenAmount, uint256 agentTokenId)
    {
        address creator = msg.sender;

        if (address(agents) == address(0)) revert AgentsNotSet();
        if (address(locker) == address(0)) revert LockerNotSet();
        if (address(tokenDeployer) == address(0)) revert DeployerNotSet();

        address paired = p.pairedToken == address(0) ? weth : p.pairedToken;
        if (!allowedPairs[paired]) revert PairNotAllowed(paired);
        if (!_isValidTick(paired, p.startingTick))
            revert InvalidStartingTick(p.startingTick, paired);

        if (bytes(p.twitter).length  > 50) revert StringTooLong("twitter");
        if (bytes(p.telegram).length > 50) revert StringTooLong("telegram");
        if (bytes(p.website).length  > 50) revert StringTooLong("website");

        // ── 1. Mint Agent NFT ─────────────────────────────────────────────
        agentTokenId = agents.mintFromFactory(creator);

        // ── 2. Deploy token ───────────────────────────────────────────────
        token = tokenDeployer.deploy(p.name, p.symbol, STANDARD_SUPPLY, address(poolManager), creator);

        // ── 3. Capture next NFT id ────────────────────────────────────────
        positionId = positionManager.nextTokenId();

        // ── 4. Compute ticks ──────────────────────────────────────────────
        bool pairedIsC0 = paired < token;
        int24 sTick     = int24(uint24(p.startingTick));

        int24 tickLower;
        int24 tickUpper;
        int24 actualStartingTick;

        if (pairedIsC0) {
            tickUpper          = sTick;
            tickLower          = sTick - TICK_RANGE;
            actualStartingTick = sTick;
        } else {
            tickLower          = -sTick;
            tickUpper          = -sTick + TICK_RANGE;
            actualStartingTick = tickLower;
        }

        uint160 sqrtPriceX96 = TickMath.getSqrtPriceAtTick(actualStartingTick);

        // ── 5. Sort currencies ────────────────────────────────────────────
        Currency currency0 = Currency.wrap(pairedIsC0 ? paired : token);
        Currency currency1 = Currency.wrap(pairedIsC0 ? token  : paired);

        PoolKey memory key = PoolKey({
            currency0:   currency0,
            currency1:   currency1,
            fee:         FIXED_FEE,
            tickSpacing: TICK_SPACING,
            hooks:       IHooks(address(0))
        });

        // ── 5b. Register token ────────────────────────────────────────────
        _tokens.push(TokenInfo({
            token:        token,
            creator:      creator,
            deployedAt:   block.timestamp,
            positionId:   positionId,
            paired:       paired,
            poolId:       key.toId(),
            fee:          FIXED_FEE,
            name:         p.name,
            symbol:       p.symbol,
            twitter:      p.twitter,
            telegram:     p.telegram,
            website:      p.website,
            agentTokenId: agentTokenId
        }));
        tokenIndex[token]            = _tokens.length;
        tokenByAgentId[agentTokenId] = token;

        // ── 6. Transfer tokens to PositionManager ─────────────────────────
        IERC20(token).transfer(address(positionManager), STANDARD_SUPPLY);

        // ── 7. Build V4 actions ───────────────────────────────────────────
        uint128 amount0Max = pairedIsC0 ? 0                        : uint128(STANDARD_SUPPLY);
        uint128 amount1Max = pairedIsC0 ? uint128(STANDARD_SUPPLY) : 0;

        bytes memory actions = abi.encodePacked(
            uint8(Actions.SETTLE),
            uint8(Actions.MINT_POSITION_FROM_DELTAS),
            uint8(Actions.CLOSE_CURRENCY)
        );

        bytes[] memory actParams = new bytes[](3);

        uint256 CONTRACT_BALANCE = 0x8000000000000000000000000000000000000000000000000000000000000000;
        Currency tokenCurrency = Currency.wrap(token);
        actParams[0] = abi.encode(tokenCurrency, CONTRACT_BALANCE, false);

        actParams[1] = abi.encode(
            key,
            tickLower,
            tickUpper,
            amount0Max,
            amount1Max,
            address(this),
            bytes("")
        );

        actParams[2] = abi.encode(tokenCurrency);

        // ── 8. Batch: initializePool + modifyLiquidities ──────────────────
        bytes[] memory calls = new bytes[](2);
        calls[0] = abi.encodeCall(IPoolInitializer_v4.initializePool, (key, sqrtPriceX96));
        calls[1] = abi.encodeCall(
            IPositionManagerMinimal.modifyLiquidities,
            (abi.encode(actions, actParams), block.timestamp)
        );

        positionManager.multicall(calls);

        // ── 9. Lock LP + register project agent ───────────────────────────
        IERC721(address(positionManager)).safeTransferFrom(
            address(this),
            address(locker),
            positionId
        );

        locker.setProjectAgent(positionId, paired, address(agents), agentTokenId);

        emit TokenDeployed(
            token,
            creator,
            paired,
            positionId,
            key.toId(),
            FIXED_FEE,
            p.name,
            p.symbol,
            p.twitter,
            p.telegram,
            p.website,
            agentTokenId,
            block.timestamp
        );

        // ── 10. Deployment fee + optional dev buy ─────────────────────────
        if (msg.value < deploymentFee) revert InsufficientDeploymentFee();

        uint256 devBuyEth = msg.value - deploymentFee;

        if (devBuyEth > 0 || p.devBuyAmountIn > 0) {
            if (devBuyEth > 0 && p.devBuyAmountIn > 0) revert DevBuyAmbiguousInput();
            if (devBuyEth > 0 && paired != weth)       revert DevBuyRequiresWETHPair();

            uint256 amountIn;
            if (devBuyEth > 0) {
                IWETH(weth).deposit{value: devBuyEth}();
                amountIn = devBuyEth;
            } else {
                IERC20(paired).transferFrom(creator, address(this), p.devBuyAmountIn);
                amountIn = p.devBuyAmountIn;
            }

            DevBuyData memory devBuy = DevBuyData({
                key:            key,
                zeroForOne:     pairedIsC0,
                amountIn:       amountIn,
                pairedCurrency: Currency.wrap(paired),
                recipient:      creator
            });

            bytes memory unlockResult = poolManager.unlock(abi.encode(devBuy));
            tokenAmount = abi.decode(unlockResult, (uint256));
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  Unlock callback — dev buy swap
    // ─────────────────────────────────────────────────────────────

    function unlockCallback(bytes calldata data) external returns (bytes memory) {
        require(msg.sender == address(poolManager), "only PM");

        DevBuyData memory d = abi.decode(data, (DevBuyData));

        BalanceDelta delta = poolManager.swap(
            d.key,
            IPoolManagerMinimal.SwapParams({
                zeroForOne:        d.zeroForOne,
                amountSpecified:   -int256(d.amountIn),
                sqrtPriceLimitX96: d.zeroForOne
                    ? TickMath.MIN_SQRT_PRICE + 1
                    : TickMath.MAX_SQRT_PRICE - 1
            }),
            bytes("")
        );

        poolManager.sync(d.pairedCurrency);
        IERC20(Currency.unwrap(d.pairedCurrency)).transfer(address(poolManager), d.amountIn);
        poolManager.settle();

        int128 tokenDelta = d.zeroForOne
            ? BalanceDeltaLibrary.amount1(delta)
            : BalanceDeltaLibrary.amount0(delta);

        uint256 tokenAmount;
        if (tokenDelta > 0) {
            tokenAmount = uint128(tokenDelta);
            Currency tokenCurrency = d.zeroForOne ? d.key.currency1 : d.key.currency0;
            poolManager.take(tokenCurrency, d.recipient, uint128(tokenDelta));
        }

        return abi.encode(tokenAmount);
    }

    // ─────────────────────────────────────────────────────────────
    //  Views
    // ─────────────────────────────────────────────────────────────

    function totalDeployed() external view returns (uint256) {
        return _tokens.length;
    }

    function getTokenByIndex(uint256 index) external view returns (TokenInfo memory) {
        return _tokens[index];
    }

    function getTokenInfo(address token) external view returns (TokenInfo memory) {
        uint256 idx = tokenIndex[token];
        if (idx == 0) revert TokenNotFound(token);
        return _tokens[idx - 1];
    }

    function getTokensPaginated(uint256 offset, uint256 limit)
        external
        view
        returns (TokenInfo[] memory page)
    {
        uint256 total = _tokens.length;
        if (offset >= total) return page;
        uint256 end = offset + limit;
        if (end > total) end = total;
        page = new TokenInfo[](end - offset);
        for (uint256 i = offset; i < end; i++) {
            page[i - offset] = _tokens[i];
        }
    }

    function getTokenInfoByAgentId(uint256 agentTokenId) external view returns (TokenInfo memory) {
        address token = tokenByAgentId[agentTokenId];
        if (token == address(0)) revert TokenNotFound(address(0));
        return _tokens[tokenIndex[token] - 1];
    }

    function isDeployed(address token) external view returns (bool) {
        return tokenIndex[token] != 0;
    }

    function getPoolKey(address token, address pairedToken) external view returns (PoolKey memory) {
        address paired    = pairedToken == address(0) ? weth : pairedToken;
        bool    pairedIsC0 = paired < token;
        return PoolKey({
            currency0:   Currency.wrap(pairedIsC0 ? paired : token),
            currency1:   Currency.wrap(pairedIsC0 ? token  : paired),
            fee:         FIXED_FEE,
            tickSpacing: TICK_SPACING,
            hooks:       IHooks(address(0))
        });
    }

    // ─────────────────────────────────────────────────────────────
    //  Deployment fee
    // ─────────────────────────────────────────────────────────────

    function setDeploymentFee(uint256 fee) external onlyOwner {
        deploymentFee = fee;
        emit DeploymentFeeSet(fee);
    }

    function withdrawETH(address to, uint256 amount) external onlyOwner {
        if (to == address(0)) revert ZeroAddress();
        (bool ok, ) = to.call{value: amount}("");
        require(ok, "ETH transfer failed");
        emit ETHWithdrawn(to, amount);
    }

    function rescueERC20(address tokenAddr, uint256 amount) external onlyOwner {
        IERC20(tokenAddr).transfer(owner(), amount);
    }

    receive() external payable {}
}


// ===== FILE: contracts/main/SatLocker.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

// Deploy your own ERC-20 token with SAT Protocol.
// Built on Uniswap V4.
// SAT Protocol introduces a structured launch framework designed around transparency, security, and immutable deployment standards.
//
// https://x.com/satprotocol_io
// https://satprotocol.io/
// https://t.me/satprotocol_io

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721Receiver.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

import {PoolKey}      from "@uniswap/v4-core/src/types/PoolKey.sol";
import {Currency}     from "@uniswap/v4-core/src/types/Currency.sol";
import {BalanceDelta} from "@uniswap/v4-core/src/types/BalanceDelta.sol";
import {IHooks}       from "@uniswap/v4-core/src/interfaces/IHooks.sol";
import {TickMath}     from "@uniswap/v4-core/src/libraries/TickMath.sol";
import {IUnlockCallback} from "@uniswap/v4-core/src/interfaces/callback/IUnlockCallback.sol";

import {IPositionManagerMinimal} from "./interfaces/IPositionManagerMinimal.sol";
import {IPoolManagerMinimal}     from "./interfaces/IPoolManagerMinimal.sol";

interface IGoPlusLocker {
    function lockNFTPosition(
        uint256 tokenId,
        uint256 unlockTime,
        address lockOwner_,
        address collector_,
        string  memory feeName
    ) external payable returns (uint256 lockId);

    function collect(
        uint256 lockId,
        address recipient
    ) external returns (uint256 amount0, uint256 amount1, uint256 fee0, uint256 fee1);

    function relock(uint256 lockId_, uint256 endTime_) external;
    function unlockLiquidity(uint256 lockId) external;
}

interface ISwapRouterV3 {
    struct ExactInputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24  fee;
        address recipient;
        uint256 amountIn;
        uint256 amountOutMinimum;
        uint160 sqrtPriceLimitX96;
    }
    function exactInputSingle(ExactInputSingleParams calldata params) external returns (uint256 amountOut);
}

interface IAgents {
    function ownerOf(uint256 tokenId) external view returns (address);
}

/**
 * @title  SatLocker
 * @notice Middleware locker for Sat Protocol LP positions.
 *
 *         Flow:
 *           1. SatFactory mints the V4 LP NFT directly to this contract.
 *           2. onERC721Received fires → NFT is forwarded to GoPlusLocker.
 *           3. Factory calls setProjectAgent() to bind the position to an Agent NFT.
 *           4. Anyone can call collectFees() — fees are converted to WETH then split:
 *                • platform wallet  — platformFeePercentage(FIXED_FEE) of total (50%)
 *                • Agent NFT owner  — remaining 50%
 */
contract SatLocker is Ownable, ReentrancyGuard, IERC721Receiver, IUnlockCallback {

    // ─────────────────────────────────────────────────────────────
    //  Constants
    // ─────────────────────────────────────────────────────────────

    uint256 public constant LOCK_DURATION = 750 days;
    string  public constant FEE_NAME      = "LVP";

    // ─────────────────────────────────────────────────────────────
    //  Types
    // ─────────────────────────────────────────────────────────────

    struct PositionData {
        uint256 lockId;
        bool    locked;
        address pairedToken;
        address agentsContract;
        uint256 agentTokenId;
    }

    struct V3RouteConfig {
        address router;
        uint24  fee;
    }

    struct TokenSwapData {
        PoolKey key;
        bool    tokenIsC0;
        uint256 amountIn;
    }

    // ─────────────────────────────────────────────────────────────
    //  State
    // ─────────────────────────────────────────────────────────────

    IGoPlusLocker           public immutable goPlusLocker;
    IPositionManagerMinimal public immutable positionManager;
    IPoolManagerMinimal     public immutable poolManager;
    address                 public immutable weth;

    address public platformWallet;
    address public platformWallet2;
    uint16  public platformWallet2Bps;

    mapping(address => V3RouteConfig) public wethRoutes;
    mapping(uint256 => PositionData)  private _positions;
    mapping(address => bool)          public authorizedCallers;

    // ─────────────────────────────────────────────────────────────
    //  Events
    // ─────────────────────────────────────────────────────────────

    event NFTLocked         (uint256 indexed nftId, uint256 lockId, uint256 unlockTime);
    event FeesCollected     (uint256 indexed nftId, address indexed agentOwner, uint256 platformShare, uint256 agentShare);
    event Relocked          (uint256 indexed nftId, uint256 newUnlockTime);
    event Unlocked          (uint256 indexed nftId);
    event NFTWithdrawn      (uint256 indexed nftId, address to);
    event ProjectAgentSet   (uint256 indexed nftId, address agentsContract, uint256 agentTokenId);
    event AuthorizedCaller  (address indexed caller, bool authorized);
    event WethRouteSet      (address indexed pairedToken, address router, uint24 fee);
    event PlatformWalletSet (address indexed wallet);
    event PlatformWallet2Set(address indexed wallet, uint16 bps);

    // ─────────────────────────────────────────────────────────────
    //  Errors
    // ─────────────────────────────────────────────────────────────

    error OnlyPositionManager();
    error OnlyPoolManager();
    error PositionNotFound(uint256 nftId);
    error NFTNotAvailable(uint256 nftId);
    error NotAuthorized();
    error NoWethRoute(address pairedToken);
    error ZeroAddress();
    error BpsExceeds10000();

    // ─────────────────────────────────────────────────────────────
    //  Modifiers
    // ─────────────────────────────────────────────────────────────

    modifier onlyAuthorized() {
        if (!authorizedCallers[msg.sender] && msg.sender != owner())
            revert NotAuthorized();
        _;
    }

    // ─────────────────────────────────────────────────────────────
    //  Constructor
    // ─────────────────────────────────────────────────────────────

    constructor(
        address goPlusLocker_,
        address positionManager_,
        address poolManager_,
        address weth_,
        address platformWallet_,
        address platformWallet2_,
        uint16  platformWallet2Bps_
    ) Ownable(msg.sender) {
        if (platformWallet_ == address(0)) revert ZeroAddress();
        if (platformWallet2Bps_ > 10_000) revert BpsExceeds10000();
        goPlusLocker       = IGoPlusLocker(goPlusLocker_);
        positionManager    = IPositionManagerMinimal(positionManager_);
        poolManager        = IPoolManagerMinimal(poolManager_);
        weth               = weth_;
        platformWallet     = platformWallet_;
        platformWallet2    = platformWallet2_;
        platformWallet2Bps = platformWallet2Bps_;
    }

    // ─────────────────────────────────────────────────────────────
    //  Platform fee formula
    // ─────────────────────────────────────────────────────────────

    /**
     * @dev Platform's share in basis points.
     *      With FIXED_FEE = 10_000 (1%), this always returns 5000 (50%).
     */
    function platformFeePercentage(uint24 fee) public pure returns (uint16) {
        if (fee == 0)      return 0;
        if (fee <= 10_000) return 5_000;
        return uint16(50_000_000 / fee);
    }

    // ─────────────────────────────────────────────────────────────
    //  ERC-721 receiver
    // ─────────────────────────────────────────────────────────────

    function onERC721Received(
        address,
        address,
        uint256 tokenId,
        bytes calldata
    ) external override returns (bytes4) {
        if (msg.sender != address(positionManager)) revert OnlyPositionManager();

        PositionData storage pos = _positions[tokenId];

        if (pos.lockId != 0 && !pos.locked) {
            emit Unlocked(tokenId);
            return this.onERC721Received.selector;
        }

        IERC721(address(positionManager)).approve(address(goPlusLocker), tokenId);

        uint256 unlockTime = block.timestamp + LOCK_DURATION;
        uint256 lockId = goPlusLocker.lockNFTPosition(
            tokenId,
            unlockTime,
            address(this),
            address(this),
            FEE_NAME
        );

        pos.lockId = lockId;
        pos.locked = true;

        emit NFTLocked(tokenId, lockId, unlockTime);
        return this.onERC721Received.selector;
    }

    // ─────────────────────────────────────────────────────────────
    //  Project agent registration
    // ─────────────────────────────────────────────────────────────

    function setProjectAgent(
        uint256 nftId,
        address pairedToken,
        address agentsContract,
        uint256 agentTokenId
    ) external onlyAuthorized {
        if (agentsContract == address(0)) revert ZeroAddress();

        PositionData storage pos = _positions[nftId];
        pos.pairedToken    = pairedToken;
        pos.agentsContract = agentsContract;
        pos.agentTokenId   = agentTokenId;

        emit ProjectAgentSet(nftId, agentsContract, agentTokenId);
    }

    // ─────────────────────────────────────────────────────────────
    //  Collect fees
    // ─────────────────────────────────────────────────────────────

    /**
     * @notice Collect accrued LP fees, convert everything to WETH, split 50/50:
     *         platform wallet and Agent NFT owner.
     */
    function collectFees(uint256 nftId) external nonReentrant {
        PositionData storage pos = _positions[nftId];
        if (pos.lockId == 0) revert PositionNotFound(nftId);

        // ── 1. Collect to this contract ───────────────────────────────────
        (uint256 amount0, uint256 amount1,,) =
            goPlusLocker.collect(pos.lockId, address(this));

        if (amount0 == 0 && amount1 == 0) return;

        (PoolKey memory poolKey,) = positionManager.getPoolAndPositionInfo(nftId);

        address c0     = Currency.unwrap(poolKey.currency0);
        address paired = pos.pairedToken;

        bool pairedIsC0    = (c0 == paired);
        uint256 tokenFees  = pairedIsC0 ? amount1 : amount0;
        uint256 pairedFees = pairedIsC0 ? amount0 : amount1;

        // ── 2. Swap launched token fees → paired (V4, same pool) ─────────
        if (tokenFees > 0) {
            bool tokenIsC0 = !pairedIsC0;
            uint256 received = _swapTokenToPaired(poolKey, tokenIsC0, tokenFees);
            pairedFees += received;
        }

        if (pairedFees == 0) return;

        // ── 3. Swap paired → WETH (V3, skip if paired is WETH) ───────────
        uint256 wethAmount;
        if (paired == weth) {
            wethAmount = pairedFees;
        } else {
            V3RouteConfig memory route = wethRoutes[paired];
            if (route.router == address(0)) revert NoWethRoute(paired);

            IERC20(paired).approve(route.router, pairedFees);
            wethAmount = ISwapRouterV3(route.router).exactInputSingle(
                ISwapRouterV3.ExactInputSingleParams({
                    tokenIn:           paired,
                    tokenOut:          weth,
                    fee:               route.fee,
                    recipient:         address(this),
                    amountIn:          pairedFees,
                    amountOutMinimum:  0,
                    sqrtPriceLimitX96: 0
                })
            );
        }

        if (wethAmount == 0) return;

        // ── 4. Two-way split: platform + agent owner ──────────────────────
        uint16  platPct       = platformFeePercentage(poolKey.fee);
        uint256 platformShare = (wethAmount * platPct) / 10_000;
        uint256 agentShare    = wethAmount - platformShare;

        address agentOwner = IAgents(pos.agentsContract).ownerOf(pos.agentTokenId);

        if (platformShare > 0) {
            if (platformWallet2 != address(0) && platformWallet2Bps > 0) {
                uint256 w2 = (platformShare * platformWallet2Bps) / 10_000;
                uint256 w1 = platformShare - w2;
                if (w1 > 0) IERC20(weth).transfer(platformWallet,  w1);
                if (w2 > 0) IERC20(weth).transfer(platformWallet2, w2);
            } else {
                IERC20(weth).transfer(platformWallet, platformShare);
            }
        }

        if (agentShare > 0) IERC20(weth).transfer(agentOwner, agentShare);

        emit FeesCollected(nftId, agentOwner, platformShare, agentShare);
    }

    // ─────────────────────────────────────────────────────────────
    //  V4 unlock callback — token → paired swap
    // ─────────────────────────────────────────────────────────────

    function _swapTokenToPaired(
        PoolKey memory key,
        bool           tokenIsC0,
        uint256        amountIn
    ) internal returns (uint256 amountOut) {
        bytes memory result = poolManager.unlock(
            abi.encode(TokenSwapData({ key: key, tokenIsC0: tokenIsC0, amountIn: amountIn }))
        );
        return abi.decode(result, (uint256));
    }

    function unlockCallback(bytes calldata data) external override returns (bytes memory) {
        if (msg.sender != address(poolManager)) revert OnlyPoolManager();

        TokenSwapData memory d = abi.decode(data, (TokenSwapData));
        bool zeroForOne = d.tokenIsC0;

        BalanceDelta delta = poolManager.swap(
            d.key,
            IPoolManagerMinimal.SwapParams({
                zeroForOne:        zeroForOne,
                amountSpecified:   -int256(d.amountIn),
                sqrtPriceLimitX96: zeroForOne
                    ? TickMath.MIN_SQRT_PRICE + 1
                    : TickMath.MAX_SQRT_PRICE - 1
            }),
            ""
        );

        Currency tokenCurrency = zeroForOne ? d.key.currency0 : d.key.currency1;
        int128   tokenDelta    = zeroForOne ? delta.amount0() : delta.amount1();
        if (tokenDelta < 0) {
            poolManager.sync(tokenCurrency);
            IERC20(Currency.unwrap(tokenCurrency)).transfer(
                address(poolManager),
                uint256(uint128(-tokenDelta))
            );
            poolManager.settle();
        }

        Currency pairedCurrency = zeroForOne ? d.key.currency1 : d.key.currency0;
        int128   pairedDelta    = zeroForOne ? delta.amount1() : delta.amount0();
        uint256  amountOut      = 0;
        if (pairedDelta > 0) {
            amountOut = uint256(uint128(pairedDelta));
            poolManager.take(pairedCurrency, address(this), amountOut);
        }

        return abi.encode(amountOut);
    }

    // ─────────────────────────────────────────────────────────────
    //  Relock / Unlock / Withdraw
    // ─────────────────────────────────────────────────────────────

    function relock(uint256 nftId, uint256 newUnlockTime) external onlyOwner {
        PositionData storage pos = _positions[nftId];
        if (pos.lockId == 0) revert PositionNotFound(nftId);
        goPlusLocker.relock(pos.lockId, newUnlockTime);
        pos.locked = true;
        emit Relocked(nftId, newUnlockTime);
    }

    function unlockLiquidity(uint256 nftId) external onlyOwner {
        PositionData storage pos = _positions[nftId];
        if (pos.lockId == 0) revert PositionNotFound(nftId);
        pos.locked = false;
        goPlusLocker.unlockLiquidity(pos.lockId);
    }

    function withdrawNFT(uint256 nftId, address to) external onlyOwner {
        if (IERC721(address(positionManager)).ownerOf(nftId) != address(this))
            revert NFTNotAvailable(nftId);
        if (_positions[nftId].locked)
            revert NFTNotAvailable(nftId);
        IERC721(address(positionManager)).safeTransferFrom(address(this), to, nftId);
        emit NFTWithdrawn(nftId, to);
    }

    // ─────────────────────────────────────────────────────────────
    //  Admin
    // ─────────────────────────────────────────────────────────────

    function setAuthorizedCaller(address caller, bool authorized) external onlyOwner {
        authorizedCallers[caller] = authorized;
        emit AuthorizedCaller(caller, authorized);
    }

    function setPlatformWallet(address wallet_) external onlyOwner {
        if (wallet_ == address(0)) revert ZeroAddress();
        platformWallet = wallet_;
        emit PlatformWalletSet(wallet_);
    }

    function setPlatformWallet2(address wallet_, uint16 bps_) external onlyOwner {
        if (bps_ > 10_000) revert BpsExceeds10000();
        platformWallet2    = wallet_;
        platformWallet2Bps = bps_;
        emit PlatformWallet2Set(wallet_, bps_);
    }

    function setWethRoute(address pairedToken, address router, uint24 fee) external onlyOwner {
        wethRoutes[pairedToken] = V3RouteConfig({ router: router, fee: fee });
        emit WethRouteSet(pairedToken, router, fee);
    }

    function rescueETH(address to, uint256 amount) external onlyOwner {
        if (to == address(0)) revert ZeroAddress();
        (bool ok,) = to.call{value: amount}("");
        require(ok, "ETH transfer failed");
    }

    function rescueERC20(address token, address to, uint256 amount) external onlyOwner {
        if (to == address(0)) revert ZeroAddress();
        IERC20(token).transfer(to, amount);
    }

    // ─────────────────────────────────────────────────────────────
    //  Views
    // ─────────────────────────────────────────────────────────────

    function getLockId(uint256 nftId) external view returns (uint256) {
        return _positions[nftId].lockId;
    }

    function isLocked(uint256 nftId) external view returns (bool) {
        return _positions[nftId].locked;
    }

    function getPairedToken(uint256 nftId) external view returns (address) {
        return _positions[nftId].pairedToken;
    }

    function getProjectAgent(uint256 nftId) external view returns (
        address agentsContract,
        uint256 agentTokenId
    ) {
        PositionData storage pos = _positions[nftId];
        return (pos.agentsContract, pos.agentTokenId);
    }

    function getAgentOwner(uint256 nftId) external view returns (address) {
        PositionData storage pos = _positions[nftId];
        if (pos.agentsContract == address(0)) return address(0);
        return IAgents(pos.agentsContract).ownerOf(pos.agentTokenId);
    }
}
