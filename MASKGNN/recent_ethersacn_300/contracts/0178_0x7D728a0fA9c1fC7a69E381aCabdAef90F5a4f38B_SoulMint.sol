// ===== FILE: _openzeppelin/contracts/access/Ownable.sol =====
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


// ===== FILE: _openzeppelin/contracts/interfaces/IERC2981.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.0) (interfaces/IERC2981.sol)

pragma solidity ^0.8.0;

import "../utils/introspection/IERC165.sol";

/**
 * @dev Interface for the NFT Royalty Standard.
 *
 * A standardized way to retrieve royalty payment information for non-fungible tokens (NFTs) to enable universal
 * support for royalty payments across all NFT marketplaces and ecosystem participants.
 *
 * _Available since v4.5._
 */
interface IERC2981 is IERC165 {
    /**
     * @dev Returns how much royalty is owed and to whom, based on a sale price that may be denominated in any unit of
     * exchange. The royalty amount is denominated and should be paid in that same unit of exchange.
     */
    function royaltyInfo(
        uint256 tokenId,
        uint256 salePrice
    ) external view returns (address receiver, uint256 royaltyAmount);
}


// ===== FILE: _openzeppelin/contracts/security/ReentrancyGuard.sol =====
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


// ===== FILE: _openzeppelin/contracts/token/ERC721/ERC721.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.0) (token/ERC721/ERC721.sol)

pragma solidity ^0.8.0;

import "./IERC721.sol";
import "./IERC721Receiver.sol";
import "./extensions/IERC721Metadata.sol";
import "../../utils/Address.sol";
import "../../utils/Context.sol";
import "../../utils/Strings.sol";
import "../../utils/introspection/ERC165.sol";

/**
 * @dev Implementation of https://eips.ethereum.org/EIPS/eip-721[ERC721] Non-Fungible Token Standard, including
 * the Metadata extension, but not including the Enumerable extension, which is available separately as
 * {ERC721Enumerable}.
 */
contract ERC721 is Context, ERC165, IERC721, IERC721Metadata {
    using Address for address;
    using Strings for uint256;

    // Token name
    string private _name;

    // Token symbol
    string private _symbol;

    // Mapping from token ID to owner address
    mapping(uint256 => address) private _owners;

    // Mapping owner address to token count
    mapping(address => uint256) private _balances;

    // Mapping from token ID to approved address
    mapping(uint256 => address) private _tokenApprovals;

    // Mapping from owner to operator approvals
    mapping(address => mapping(address => bool)) private _operatorApprovals;

    /**
     * @dev Initializes the contract by setting a `name` and a `symbol` to the token collection.
     */
    constructor(string memory name_, string memory symbol_) {
        _name = name_;
        _symbol = symbol_;
    }

    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId) public view virtual override(ERC165, IERC165) returns (bool) {
        return
            interfaceId == type(IERC721).interfaceId ||
            interfaceId == type(IERC721Metadata).interfaceId ||
            super.supportsInterface(interfaceId);
    }

    /**
     * @dev See {IERC721-balanceOf}.
     */
    function balanceOf(address owner) public view virtual override returns (uint256) {
        require(owner != address(0), "ERC721: address zero is not a valid owner");
        return _balances[owner];
    }

    /**
     * @dev See {IERC721-ownerOf}.
     */
    function ownerOf(uint256 tokenId) public view virtual override returns (address) {
        address owner = _ownerOf(tokenId);
        require(owner != address(0), "ERC721: invalid token ID");
        return owner;
    }

    /**
     * @dev See {IERC721Metadata-name}.
     */
    function name() public view virtual override returns (string memory) {
        return _name;
    }

    /**
     * @dev See {IERC721Metadata-symbol}.
     */
    function symbol() public view virtual override returns (string memory) {
        return _symbol;
    }

    /**
     * @dev See {IERC721Metadata-tokenURI}.
     */
    function tokenURI(uint256 tokenId) public view virtual override returns (string memory) {
        _requireMinted(tokenId);

        string memory baseURI = _baseURI();
        return bytes(baseURI).length > 0 ? string(abi.encodePacked(baseURI, tokenId.toString())) : "";
    }

    /**
     * @dev Base URI for computing {tokenURI}. If set, the resulting URI for each
     * token will be the concatenation of the `baseURI` and the `tokenId`. Empty
     * by default, can be overridden in child contracts.
     */
    function _baseURI() internal view virtual returns (string memory) {
        return "";
    }

    /**
     * @dev See {IERC721-approve}.
     */
    function approve(address to, uint256 tokenId) public virtual override {
        address owner = ERC721.ownerOf(tokenId);
        require(to != owner, "ERC721: approval to current owner");

        require(
            _msgSender() == owner || isApprovedForAll(owner, _msgSender()),
            "ERC721: approve caller is not token owner or approved for all"
        );

        _approve(to, tokenId);
    }

    /**
     * @dev See {IERC721-getApproved}.
     */
    function getApproved(uint256 tokenId) public view virtual override returns (address) {
        _requireMinted(tokenId);

        return _tokenApprovals[tokenId];
    }

    /**
     * @dev See {IERC721-setApprovalForAll}.
     */
    function setApprovalForAll(address operator, bool approved) public virtual override {
        _setApprovalForAll(_msgSender(), operator, approved);
    }

    /**
     * @dev See {IERC721-isApprovedForAll}.
     */
    function isApprovedForAll(address owner, address operator) public view virtual override returns (bool) {
        return _operatorApprovals[owner][operator];
    }

    /**
     * @dev See {IERC721-transferFrom}.
     */
    function transferFrom(address from, address to, uint256 tokenId) public virtual override {
        //solhint-disable-next-line max-line-length
        require(_isApprovedOrOwner(_msgSender(), tokenId), "ERC721: caller is not token owner or approved");

        _transfer(from, to, tokenId);
    }

    /**
     * @dev See {IERC721-safeTransferFrom}.
     */
    function safeTransferFrom(address from, address to, uint256 tokenId) public virtual override {
        safeTransferFrom(from, to, tokenId, "");
    }

    /**
     * @dev See {IERC721-safeTransferFrom}.
     */
    function safeTransferFrom(address from, address to, uint256 tokenId, bytes memory data) public virtual override {
        require(_isApprovedOrOwner(_msgSender(), tokenId), "ERC721: caller is not token owner or approved");
        _safeTransfer(from, to, tokenId, data);
    }

    /**
     * @dev Safely transfers `tokenId` token from `from` to `to`, checking first that contract recipients
     * are aware of the ERC721 protocol to prevent tokens from being forever locked.
     *
     * `data` is additional data, it has no specified format and it is sent in call to `to`.
     *
     * This internal function is equivalent to {safeTransferFrom}, and can be used to e.g.
     * implement alternative mechanisms to perform token transfer, such as signature-based.
     *
     * Requirements:
     *
     * - `from` cannot be the zero address.
     * - `to` cannot be the zero address.
     * - `tokenId` token must exist and be owned by `from`.
     * - If `to` refers to a smart contract, it must implement {IERC721Receiver-onERC721Received}, which is called upon a safe transfer.
     *
     * Emits a {Transfer} event.
     */
    function _safeTransfer(address from, address to, uint256 tokenId, bytes memory data) internal virtual {
        _transfer(from, to, tokenId);
        require(_checkOnERC721Received(from, to, tokenId, data), "ERC721: transfer to non ERC721Receiver implementer");
    }

    /**
     * @dev Returns the owner of the `tokenId`. Does NOT revert if token doesn't exist
     */
    function _ownerOf(uint256 tokenId) internal view virtual returns (address) {
        return _owners[tokenId];
    }

    /**
     * @dev Returns whether `tokenId` exists.
     *
     * Tokens can be managed by their owner or approved accounts via {approve} or {setApprovalForAll}.
     *
     * Tokens start existing when they are minted (`_mint`),
     * and stop existing when they are burned (`_burn`).
     */
    function _exists(uint256 tokenId) internal view virtual returns (bool) {
        return _ownerOf(tokenId) != address(0);
    }

    /**
     * @dev Returns whether `spender` is allowed to manage `tokenId`.
     *
     * Requirements:
     *
     * - `tokenId` must exist.
     */
    function _isApprovedOrOwner(address spender, uint256 tokenId) internal view virtual returns (bool) {
        address owner = ERC721.ownerOf(tokenId);
        return (spender == owner || isApprovedForAll(owner, spender) || getApproved(tokenId) == spender);
    }

    /**
     * @dev Safely mints `tokenId` and transfers it to `to`.
     *
     * Requirements:
     *
     * - `tokenId` must not exist.
     * - If `to` refers to a smart contract, it must implement {IERC721Receiver-onERC721Received}, which is called upon a safe transfer.
     *
     * Emits a {Transfer} event.
     */
    function _safeMint(address to, uint256 tokenId) internal virtual {
        _safeMint(to, tokenId, "");
    }

    /**
     * @dev Same as {xref-ERC721-_safeMint-address-uint256-}[`_safeMint`], with an additional `data` parameter which is
     * forwarded in {IERC721Receiver-onERC721Received} to contract recipients.
     */
    function _safeMint(address to, uint256 tokenId, bytes memory data) internal virtual {
        _mint(to, tokenId);
        require(
            _checkOnERC721Received(address(0), to, tokenId, data),
            "ERC721: transfer to non ERC721Receiver implementer"
        );
    }

    /**
     * @dev Mints `tokenId` and transfers it to `to`.
     *
     * WARNING: Usage of this method is discouraged, use {_safeMint} whenever possible
     *
     * Requirements:
     *
     * - `tokenId` must not exist.
     * - `to` cannot be the zero address.
     *
     * Emits a {Transfer} event.
     */
    function _mint(address to, uint256 tokenId) internal virtual {
        require(to != address(0), "ERC721: mint to the zero address");
        require(!_exists(tokenId), "ERC721: token already minted");

        _beforeTokenTransfer(address(0), to, tokenId, 1);

        // Check that tokenId was not minted by `_beforeTokenTransfer` hook
        require(!_exists(tokenId), "ERC721: token already minted");

        unchecked {
            // Will not overflow unless all 2**256 token ids are minted to the same owner.
            // Given that tokens are minted one by one, it is impossible in practice that
            // this ever happens. Might change if we allow batch minting.
            // The ERC fails to describe this case.
            _balances[to] += 1;
        }

        _owners[tokenId] = to;

        emit Transfer(address(0), to, tokenId);

        _afterTokenTransfer(address(0), to, tokenId, 1);
    }

    /**
     * @dev Destroys `tokenId`.
     * The approval is cleared when the token is burned.
     * This is an internal function that does not check if the sender is authorized to operate on the token.
     *
     * Requirements:
     *
     * - `tokenId` must exist.
     *
     * Emits a {Transfer} event.
     */
    function _burn(uint256 tokenId) internal virtual {
        address owner = ERC721.ownerOf(tokenId);

        _beforeTokenTransfer(owner, address(0), tokenId, 1);

        // Update ownership in case tokenId was transferred by `_beforeTokenTransfer` hook
        owner = ERC721.ownerOf(tokenId);

        // Clear approvals
        delete _tokenApprovals[tokenId];

        unchecked {
            // Cannot overflow, as that would require more tokens to be burned/transferred
            // out than the owner initially received through minting and transferring in.
            _balances[owner] -= 1;
        }
        delete _owners[tokenId];

        emit Transfer(owner, address(0), tokenId);

        _afterTokenTransfer(owner, address(0), tokenId, 1);
    }

    /**
     * @dev Transfers `tokenId` from `from` to `to`.
     *  As opposed to {transferFrom}, this imposes no restrictions on msg.sender.
     *
     * Requirements:
     *
     * - `to` cannot be the zero address.
     * - `tokenId` token must be owned by `from`.
     *
     * Emits a {Transfer} event.
     */
    function _transfer(address from, address to, uint256 tokenId) internal virtual {
        require(ERC721.ownerOf(tokenId) == from, "ERC721: transfer from incorrect owner");
        require(to != address(0), "ERC721: transfer to the zero address");

        _beforeTokenTransfer(from, to, tokenId, 1);

        // Check that tokenId was not transferred by `_beforeTokenTransfer` hook
        require(ERC721.ownerOf(tokenId) == from, "ERC721: transfer from incorrect owner");

        // Clear approvals from the previous owner
        delete _tokenApprovals[tokenId];

        unchecked {
            // `_balances[from]` cannot overflow for the same reason as described in `_burn`:
            // `from`'s balance is the number of token held, which is at least one before the current
            // transfer.
            // `_balances[to]` could overflow in the conditions described in `_mint`. That would require
            // all 2**256 token ids to be minted, which in practice is impossible.
            _balances[from] -= 1;
            _balances[to] += 1;
        }
        _owners[tokenId] = to;

        emit Transfer(from, to, tokenId);

        _afterTokenTransfer(from, to, tokenId, 1);
    }

    /**
     * @dev Approve `to` to operate on `tokenId`
     *
     * Emits an {Approval} event.
     */
    function _approve(address to, uint256 tokenId) internal virtual {
        _tokenApprovals[tokenId] = to;
        emit Approval(ERC721.ownerOf(tokenId), to, tokenId);
    }

    /**
     * @dev Approve `operator` to operate on all of `owner` tokens
     *
     * Emits an {ApprovalForAll} event.
     */
    function _setApprovalForAll(address owner, address operator, bool approved) internal virtual {
        require(owner != operator, "ERC721: approve to caller");
        _operatorApprovals[owner][operator] = approved;
        emit ApprovalForAll(owner, operator, approved);
    }

    /**
     * @dev Reverts if the `tokenId` has not been minted yet.
     */
    function _requireMinted(uint256 tokenId) internal view virtual {
        require(_exists(tokenId), "ERC721: invalid token ID");
    }

    /**
     * @dev Internal function to invoke {IERC721Receiver-onERC721Received} on a target address.
     * The call is not executed if the target address is not a contract.
     *
     * @param from address representing the previous owner of the given token ID
     * @param to target address that will receive the tokens
     * @param tokenId uint256 ID of the token to be transferred
     * @param data bytes optional data to send along with the call
     * @return bool whether the call correctly returned the expected magic value
     */
    function _checkOnERC721Received(
        address from,
        address to,
        uint256 tokenId,
        bytes memory data
    ) private returns (bool) {
        if (to.isContract()) {
            try IERC721Receiver(to).onERC721Received(_msgSender(), from, tokenId, data) returns (bytes4 retval) {
                return retval == IERC721Receiver.onERC721Received.selector;
            } catch (bytes memory reason) {
                if (reason.length == 0) {
                    revert("ERC721: transfer to non ERC721Receiver implementer");
                } else {
                    /// @solidity memory-safe-assembly
                    assembly {
                        revert(add(32, reason), mload(reason))
                    }
                }
            }
        } else {
            return true;
        }
    }

    /**
     * @dev Hook that is called before any token transfer. This includes minting and burning. If {ERC721Consecutive} is
     * used, the hook may be called as part of a consecutive (batch) mint, as indicated by `batchSize` greater than 1.
     *
     * Calling conditions:
     *
     * - When `from` and `to` are both non-zero, ``from``'s tokens will be transferred to `to`.
     * - When `from` is zero, the tokens will be minted for `to`.
     * - When `to` is zero, ``from``'s tokens will be burned.
     * - `from` and `to` are never both zero.
     * - `batchSize` is non-zero.
     *
     * To learn more about hooks, head to xref:ROOT:extending-contracts.adoc#using-hooks[Using Hooks].
     */
    function _beforeTokenTransfer(address from, address to, uint256 firstTokenId, uint256 batchSize) internal virtual {}

    /**
     * @dev Hook that is called after any token transfer. This includes minting and burning. If {ERC721Consecutive} is
     * used, the hook may be called as part of a consecutive (batch) mint, as indicated by `batchSize` greater than 1.
     *
     * Calling conditions:
     *
     * - When `from` and `to` are both non-zero, ``from``'s tokens were transferred to `to`.
     * - When `from` is zero, the tokens were minted for `to`.
     * - When `to` is zero, ``from``'s tokens were burned.
     * - `from` and `to` are never both zero.
     * - `batchSize` is non-zero.
     *
     * To learn more about hooks, head to xref:ROOT:extending-contracts.adoc#using-hooks[Using Hooks].
     */
    function _afterTokenTransfer(address from, address to, uint256 firstTokenId, uint256 batchSize) internal virtual {}

    /**
     * @dev Unsafe write access to the balances, used by extensions that "mint" tokens using an {ownerOf} override.
     *
     * WARNING: Anyone calling this MUST ensure that the balances remain consistent with the ownership. The invariant
     * being that for any address `a` the value returned by `balanceOf(a)` must be equal to the number of tokens such
     * that `ownerOf(tokenId)` is `a`.
     */
    // solhint-disable-next-line func-name-mixedcase
    function __unsafe_increaseBalance(address account, uint256 amount) internal {
        _balances[account] += amount;
    }
}


// ===== FILE: _openzeppelin/contracts/token/ERC721/extensions/IERC721Metadata.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (token/ERC721/extensions/IERC721Metadata.sol)

pragma solidity ^0.8.0;

import "../IERC721.sol";

/**
 * @title ERC-721 Non-Fungible Token Standard, optional metadata extension
 * @dev See https://eips.ethereum.org/EIPS/eip-721
 */
interface IERC721Metadata is IERC721 {
    /**
     * @dev Returns the token collection name.
     */
    function name() external view returns (string memory);

    /**
     * @dev Returns the token collection symbol.
     */
    function symbol() external view returns (string memory);

    /**
     * @dev Returns the Uniform Resource Identifier (URI) for `tokenId` token.
     */
    function tokenURI(uint256 tokenId) external view returns (string memory);
}


// ===== FILE: _openzeppelin/contracts/token/ERC721/IERC721.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.0) (token/ERC721/IERC721.sol)

pragma solidity ^0.8.0;

import "../../utils/introspection/IERC165.sol";

/**
 * @dev Required interface of an ERC721 compliant contract.
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
     * - If `to` refers to a smart contract, it must implement {IERC721Receiver-onERC721Received}, which is called upon a safe transfer.
     *
     * Emits a {Transfer} event.
     */
    function safeTransferFrom(address from, address to, uint256 tokenId, bytes calldata data) external;

    /**
     * @dev Safely transfers `tokenId` token from `from` to `to`, checking first that contract recipients
     * are aware of the ERC721 protocol to prevent tokens from being forever locked.
     *
     * Requirements:
     *
     * - `from` cannot be the zero address.
     * - `to` cannot be the zero address.
     * - `tokenId` token must exist and be owned by `from`.
     * - If the caller is not `from`, it must have been allowed to move this token by either {approve} or {setApprovalForAll}.
     * - If `to` refers to a smart contract, it must implement {IERC721Receiver-onERC721Received}, which is called upon a safe transfer.
     *
     * Emits a {Transfer} event.
     */
    function safeTransferFrom(address from, address to, uint256 tokenId) external;

    /**
     * @dev Transfers `tokenId` token from `from` to `to`.
     *
     * WARNING: Note that the caller is responsible to confirm that the recipient is capable of receiving ERC721
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
     * - The `operator` cannot be the caller.
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
// OpenZeppelin Contracts (last updated v4.6.0) (token/ERC721/IERC721Receiver.sol)

pragma solidity ^0.8.0;

/**
 * @title ERC721 token receiver interface
 * @dev Interface for any contract that wants to support safeTransfers
 * from ERC721 asset contracts.
 */
interface IERC721Receiver {
    /**
     * @dev Whenever an {IERC721} `tokenId` token is transferred to this contract via {IERC721-safeTransferFrom}
     * by `operator` from `from`, this function is called.
     *
     * It must return its Solidity selector to confirm the token transfer.
     * If any other value is returned or the interface is not implemented by the recipient, the transfer will be reverted.
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


// ===== FILE: _openzeppelin/contracts/utils/Address.sol =====
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


// ===== FILE: _openzeppelin/contracts/utils/Base64.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.6) (utils/Base64.sol)

pragma solidity ^0.8.0;

/**
 * @dev Provides a set of functions to operate with Base64 strings.
 *
 * _Available since v4.5._
 */
library Base64 {
    /**
     * @dev Base64 Encoding/Decoding Table
     */
    string internal constant _TABLE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    /**
     * @dev Converts a `bytes` to its Bytes64 `string` representation.
     */
    function encode(bytes memory data) internal pure returns (string memory) {
        /**
         * Inspired by Brecht Devos (Brechtpd) implementation - MIT licence
         * https://github.com/Brechtpd/base64/blob/e78d9fd951e7b0977ddca77d92dc85183770daf4/base64.sol
         */
        if (data.length == 0) return "";

        // Loads the table into memory
        string memory table = _TABLE;

        // Encoding takes 3 bytes chunks of binary data from `bytes` data parameter
        // and split into 4 numbers of 6 bits.
        // The final Base64 length should be `bytes` data length multiplied by 4/3 rounded up
        // - `data.length + 2`  -> Round up
        // - `/ 3`              -> Number of 3-bytes chunks
        // - `4 *`              -> 4 characters for each chunk
        string memory result = new string(4 * ((data.length + 2) / 3));

        /// @solidity memory-safe-assembly
        assembly {
            // Prepare the lookup table (skip the first "length" byte)
            let tablePtr := add(table, 1)

            // Prepare result pointer, jump over length
            let resultPtr := add(result, 0x20)
            let dataPtr := data
            let endPtr := add(data, mload(data))

            // In some cases, the last iteration will read bytes after the end of the data. We cache the value, and
            // set it to zero to make sure no dirty bytes are read in that section.
            let afterPtr := add(endPtr, 0x20)
            let afterCache := mload(afterPtr)
            mstore(afterPtr, 0x00)

            // Run over the input, 3 bytes at a time
            for {

            } lt(dataPtr, endPtr) {

            } {
                // Advance 3 bytes
                dataPtr := add(dataPtr, 3)
                let input := mload(dataPtr)

                // To write each character, shift the 3 byte (24 bits) chunk
                // 4 times in blocks of 6 bits for each character (18, 12, 6, 0)
                // and apply logical AND with 0x3F to bitmask the least significant 6 bits.
                // Use this as an index into the lookup table, mload an entire word
                // so the desired character is in the least significant byte, and
                // mstore8 this least significant byte into the result and continue.

                mstore8(resultPtr, mload(add(tablePtr, and(shr(18, input), 0x3F))))
                resultPtr := add(resultPtr, 1) // Advance

                mstore8(resultPtr, mload(add(tablePtr, and(shr(12, input), 0x3F))))
                resultPtr := add(resultPtr, 1) // Advance

                mstore8(resultPtr, mload(add(tablePtr, and(shr(6, input), 0x3F))))
                resultPtr := add(resultPtr, 1) // Advance

                mstore8(resultPtr, mload(add(tablePtr, and(input, 0x3F))))
                resultPtr := add(resultPtr, 1) // Advance
            }

            // Reset the value that was cached
            mstore(afterPtr, afterCache)

            // When data `bytes` is not exactly 3 bytes long
            // it is padded with `=` characters at the end
            switch mod(mload(data), 3)
            case 1 {
                mstore8(sub(resultPtr, 1), 0x3d)
                mstore8(sub(resultPtr, 2), 0x3d)
            }
            case 2 {
                mstore8(sub(resultPtr, 1), 0x3d)
            }
        }

        return result;
    }
}


// ===== FILE: _openzeppelin/contracts/utils/Context.sol =====
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


// ===== FILE: _openzeppelin/contracts/utils/introspection/ERC165.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (utils/introspection/ERC165.sol)

pragma solidity ^0.8.0;

import "./IERC165.sol";

/**
 * @dev Implementation of the {IERC165} interface.
 *
 * Contracts that want to implement ERC165 should inherit from this contract and override {supportsInterface} to check
 * for the additional interface id that will be supported. For example:
 *
 * ```solidity
 * function supportsInterface(bytes4 interfaceId) public view virtual override returns (bool) {
 *     return interfaceId == type(MyInterface).interfaceId || super.supportsInterface(interfaceId);
 * }
 * ```
 *
 * Alternatively, {ERC165Storage} provides an easier to use but more expensive implementation.
 */
abstract contract ERC165 is IERC165 {
    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId) public view virtual override returns (bool) {
        return interfaceId == type(IERC165).interfaceId;
    }
}


// ===== FILE: _openzeppelin/contracts/utils/introspection/IERC165.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (utils/introspection/IERC165.sol)

pragma solidity ^0.8.0;

/**
 * @dev Interface of the ERC165 standard, as defined in the
 * https://eips.ethereum.org/EIPS/eip-165[EIP].
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
     * https://eips.ethereum.org/EIPS/eip-165#how-interfaces-are-identified[EIP section]
     * to learn more about how these ids are created.
     *
     * This function call must use less than 30 000 gas.
     */
    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}


// ===== FILE: _openzeppelin/contracts/utils/math/Math.sol =====
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


// ===== FILE: _openzeppelin/contracts/utils/math/SignedMath.sol =====
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


// ===== FILE: _openzeppelin/contracts/utils/Strings.sol =====
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


// ===== FILE: contracts/flatworld/FlatModel.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/utils/Strings.sol";
import "@openzeppelin/contracts/utils/Base64.sol";
import "./SoulShapes.sol";

/**
 * @title FlatModel v2.1
 * @notice On-chain SVG renderer; every NFT is visually unique.
 *         Color, geometry, and body SVG are provided by the external SoulShapes
 *         library to stay within the 24 KB contract size limit.
 *
 * Visual uniqueness sources (>9 million combinations):
 *   colorIdx = tokenId % 8        → 8 color/shape variants
 *   geomVar  = (tokenId / 8) % 5 → 5 geometry variants per shape
 *   seed     = tokenId % 99       → 99 independent texture patterns
 *   logic / creativity / morality / decisiveness → glow / eyes / gaze / expression
 */
contract FlatModel {
    using Strings for uint256;

    // ─────────────────────────────────────────────────────────────
    //  Rarity
    // ─────────────────────────────────────────────────────────────

    function _rarityLevel(uint256 l, uint256 c, uint256 m, uint256 d) private pure returns (uint8) {
        uint256 avg = (l + c + m + d) / 4;
        if (avg >= 90) return 4;
        if (avg >= 75) return 3;
        if (avg >= 60) return 2;
        if (avg >= 40) return 1;
        return 0;
    }

    function getRarity(uint256 l, uint256 c, uint256 m, uint256 d) public pure returns (string memory) {
        uint8 lv = _rarityLevel(l, c, m, d);
        if (lv == 4) return "LEGENDARY";
        if (lv == 3) return "EPIC";
        if (lv == 2) return "RARE";
        if (lv == 1) return "UNCOMMON";
        return "COMMON";
    }

    function _traitTier(uint256 v) private pure returns (uint8) {
        if (v >= 80) return 4;
        if (v >= 65) return 3;
        if (v >= 50) return 2;
        if (v >= 35) return 1;
        return 0;
    }

    function getGlowName(uint256 logic, uint256, uint256, uint256) public pure returns (string memory) {
        uint8 t = _traitTier(logic);
        if (t == 4) return "GILDED"; if (t == 3) return "RADIANT";
        if (t == 2) return "VIBRANT"; if (t == 1) return "FAINT"; return "NONE";
    }
    function getSkinName(uint256, uint256 creativity, uint256, uint256) public pure returns (string memory) {
        uint8 t = _traitTier(creativity);
        if (t == 4) return "PRISMATIC"; if (t == 3) return "LUSTROUS";
        if (t == 2) return "MARBLED"; if (t == 1) return "CLOUDY"; return "ROUGH";
    }
    function getMarkName(uint256, uint256, uint256 morality, uint256) public pure returns (string memory) {
        uint8 t = _traitTier(morality);
        if (t == 4) return "ROYAL"; if (t == 3) return "NOBLE";
        if (t == 2) return "GUILD"; if (t == 1) return "CIVIC"; return "NONE";
    }
    function getFaceName(uint256, uint256, uint256, uint256 decisiveness) public pure returns (string memory) {
        uint8 t = _traitTier(decisiveness);
        if (t == 4) return "BLISS"; if (t == 3) return "PROUD";
        if (t == 2) return "COMPOSED"; if (t == 1) return "PENSIVE"; return "WEARY";
    }

    // ─────────────────────────────────────────────────────────────
    //  Texture filter matrix (fixed per shape type, independent of colorIdx)
    // ─────────────────────────────────────────────────────────────

    function _texMatrix(bytes32 h) private pure returns (string memory) {
        if (h == keccak256("triangle")) return "0 0 0 0 0.82  0 0 0 0 1.00  0 0 0 0 0.88  0 0 0 3.0 -1.5";
        if (h == keccak256("square"))   return "0 0 0 0 0.80  0 0 0 0 0.92  0 0 0 0 1.00  0 0 0 3.0 -1.5";
        if (h == keccak256("hexagon"))  return "0 0 0 0 1.00  0 0 0 0 0.85  0 0 0 0 0.92  0 0 0 3.0 -1.5";
        return "0 0 0 0 1.00  0 0 0 0 0.90  0 0 0 0 0.55  0 0 0 3.0 -1.5";
    }

    // ─────────────────────────────────────────────────────────────
    //  Background gradient & effects (fixed per rarity)
    // ─────────────────────────────────────────────────────────────

    function _bgGradient(uint8 rarity) private pure returns (string memory) {
        if (rarity == 4) return string(abi.encodePacked(
            '<radialGradient id="sky" cx="50%" cy="48%" r="65%" gradientUnits="objectBoundingBox">',
            '<stop offset="0%" stop-color="#3d0050"/>',
            '<stop offset="55%" stop-color="#180022"/>',
            '<stop offset="100%" stop-color="#04000a"/>',
            '</radialGradient>'
        ));
        if (rarity == 3) return string(abi.encodePacked(
            '<radialGradient id="sky" cx="50%" cy="45%" r="65%" gradientUnits="objectBoundingBox">',
            '<stop offset="0%" stop-color="#6b2800"/>',
            '<stop offset="55%" stop-color="#2d0e00"/>',
            '<stop offset="100%" stop-color="#0a0300"/>',
            '</radialGradient>'
        ));
        if (rarity == 2) return string(abi.encodePacked(
            '<radialGradient id="sky" cx="50%" cy="45%" r="70%" gradientUnits="objectBoundingBox">',
            '<stop offset="0%" stop-color="#003a7a"/>',
            '<stop offset="55%" stop-color="#001845"/>',
            '<stop offset="100%" stop-color="#00091a"/>',
            '</radialGradient>'
        ));
        if (rarity == 1) return string(abi.encodePacked(
            '<linearGradient id="sky" x1="0" y1="0" x2="1" y2="1">',
            '<stop offset="0%" stop-color="#111520"/>',
            '<stop offset="100%" stop-color="#08090f"/>',
            '</linearGradient>'
        ));
        return string(abi.encodePacked(
            '<linearGradient id="sky" x1="0" y1="0" x2="1" y2="1">',
            '<stop offset="0%" stop-color="#1a1f1a"/>',
            '<stop offset="100%" stop-color="#0e120e"/>',
            '</linearGradient>'
        ));
    }

    function _getDefs(bytes32 shapeHash, uint8 rarity, uint256 tokenId) private pure returns (string memory) {
        return string(abi.encodePacked(
            '<defs>',
            '<filter id="tex" x="0%" y="0%" width="100%" height="100%" color-interpolation-filters="sRGB">',
            '<feTurbulence type="fractalNoise" baseFrequency="0.009 0.007" numOctaves="5" seed="',
            (tokenId % 99 + 1).toString(), '" result="n"/>',
            '<feColorMatrix type="luminanceToAlpha" in="n" result="la"/>',
            '<feColorMatrix type="matrix" values="', _texMatrix(shapeHash), '" in="la" result="pat"/>',
            '<feComposite in="pat" in2="SourceGraphic" operator="in"/>',
            '</filter>',
            '<filter id="halo" x="-50%" y="-50%" width="200%" height="200%">',
            '<feGaussianBlur stdDeviation="28"/>',
            '</filter>',
            _bgGradient(rarity),
            '</defs>'
        ));
    }

    function _bgLegendaryFx() private pure returns (string memory) {
        return string(abi.encodePacked(
            '<circle cx="400" cy="285" r="280" fill="rgba(255,200,80,0.07)" filter="url(#halo)"/>',
            '<circle cx="400" cy="285" r="160" fill="rgba(255,180,50,0.10)" filter="url(#halo)"/>',
            '<circle cx="400" cy="42"  r="2.8" fill="#ffd700" opacity="0.9"><animate attributeName="opacity" values="0.9;0.2;0.9" dur="2.2s" repeatCount="indefinite"/></circle>',
            '<circle cx="570" cy="110" r="2"   fill="#ffd700" opacity="0.7"><animate attributeName="opacity" values="0.7;0.15;0.7" dur="3.1s" repeatCount="indefinite"/></circle>',
            '<circle cx="680" cy="245" r="1.8" fill="#ffd700" opacity="0.6"><animate attributeName="opacity" values="0.6;0.1;0.6" dur="4.0s" repeatCount="indefinite"/></circle>',
            '<circle cx="230" cy="110" r="2"   fill="#ffd700" opacity="0.65"><animate attributeName="opacity" values="0.65;0.1;0.65" dur="3.5s" repeatCount="indefinite"/></circle>',
            '<circle cx="120" cy="245" r="1.8" fill="#ffd700" opacity="0.55"><animate attributeName="opacity" values="0.55;0.1;0.55" dur="4.8s" repeatCount="indefinite"/></circle>'
        ));
    }

    function _bgEpicFx() private pure returns (string memory) {
        return string(abi.encodePacked(
            '<circle cx="400" cy="285" r="240" fill="rgba(255,150,30,0.10)" filter="url(#halo)"/>',
            '<circle cx="400" cy="285" r="140" fill="rgba(255,120,20,0.12)" filter="url(#halo)"/>',
            '<circle cx="400" cy="70"  r="2.5" fill="#ffb347" opacity="0.8"><animate attributeName="opacity" values="0.8;0.2;0.8" dur="2.6s" repeatCount="indefinite"/></circle>',
            '<circle cx="590" cy="155" r="2"   fill="#ffb347" opacity="0.6"><animate attributeName="opacity" values="0.6;0.1;0.6" dur="3.4s" repeatCount="indefinite"/></circle>',
            '<circle cx="210" cy="155" r="2"   fill="#ffb347" opacity="0.6"><animate attributeName="opacity" values="0.6;0.1;0.6" dur="3.8s" repeatCount="indefinite"/></circle>'
        ));
    }

    function _getBackground(uint8 rarity) private pure returns (string memory) {
        string memory fx = "";
        if (rarity == 4)      fx = _bgLegendaryFx();
        else if (rarity == 3) fx = _bgEpicFx();
        else if (rarity == 2) fx = '<circle cx="400" cy="285" r="200" fill="rgba(60,130,255,0.07)" filter="url(#halo)"/>';
        return string(abi.encodePacked('<rect width="800" height="800" fill="url(#sky)"/>', fx));
    }

    // ─────────────────────────────────────────────────────────────
    //  Eye size (creativity tier) & gaze direction (morality tier)
    // ─────────────────────────────────────────────────────────────

    function _eyeScale(uint256 base, uint8 tier) private pure returns (string memory) {
        if (tier == 4) return (base * 14 / 10).toString();
        if (tier == 3) return (base * 12 / 10).toString();
        if (tier == 2) return base.toString();
        if (tier == 1) return (base * 80 / 100).toString();
        return (base * 60 / 100).toString();
    }

    function _eyeRy(uint256 base, uint8 creaTier, uint8 morTier) private pure returns (string memory) {
        uint256 scaled;
        if      (creaTier == 4) scaled = base * 14 / 10;
        else if (creaTier == 3) scaled = base * 12 / 10;
        else if (creaTier == 1) scaled = base * 80 / 100;
        else if (creaTier == 0) scaled = base * 60 / 100;
        else                    scaled = base;
        if (morTier == 4) {
            uint256 sq = scaled * 3 / 10;
            return (sq < 4 ? uint256(4) : sq).toString();
        }
        return scaled.toString();
    }

    // Circle pupil (eye center 568, 252)
    function _circlePupilCx(uint8 t) private pure returns (string memory) {
        if (t >= 3) return "568"; if (t == 2) return "563"; if (t == 1) return "562"; return "558";
    }
    function _circlePupilCy(uint8 t) private pure returns (string memory) {
        if (t >= 3) return "252"; if (t == 2) return "245"; if (t == 1) return "255"; return "263";
    }

    // Triangle pupil (eye center 547, 370)
    function _triPupilCx(uint8 t) private pure returns (string memory) {
        if (t >= 3) return "547"; if (t == 2) return "543"; if (t == 1) return "542"; return "538";
    }
    function _triPupilCy(uint8 t) private pure returns (string memory) {
        if (t >= 3) return "370"; if (t == 2) return "363"; if (t == 1) return "374"; return "381";
    }

    // Square pupil (eye center 554, 244)
    function _sqPupilCx(uint8 t) private pure returns (string memory) {
        if (t >= 3) return "554"; if (t == 2) return "550"; if (t == 1) return "549"; return "545";
    }
    function _sqPupilCy(uint8 t) private pure returns (string memory) {
        if (t >= 3) return "244"; if (t == 2) return "237"; if (t == 1) return "248"; return "255";
    }

    // Hexagon pupil (eye center 554, 232)
    function _hexPupilCx(uint8 t) private pure returns (string memory) {
        if (t >= 3) return "554"; if (t == 2) return "551"; if (t == 1) return "550"; return "546";
    }
    function _hexPupilCy(uint8 t) private pure returns (string memory) {
        if (t >= 3) return "232"; if (t == 2) return "221"; if (t == 1) return "239"; return "247";
    }

    // ─────────────────────────────────────────────────────────────
    //  Mouth expression (decisiveness tier)
    // ─────────────────────────────────────────────────────────────

    function _mouthCircle(uint8 t) private pure returns (string memory) {
        if (t == 4) return "M 528 320 Q 552 348 574 320";
        if (t == 3) return "M 534 322 Q 553 340 570 324";
        if (t == 2) return "M 536 322 Q 553 330 568 322";
        if (t == 1) return "M 536 324 L 568 324";
        return             "M 534 330 Q 553 314 570 330";
    }
    function _mouthTriangle(uint8 t) private pure returns (string memory) {
        if (t == 4) return "M 513 428 Q 532 408 553 428";
        if (t == 3) return "M 516 428 Q 533 414 550 428";
        if (t == 2) return "M 521 428 Q 533 422 545 428";
        if (t == 1) return "M 521 429 L 545 429";
        return             "M 519 435 Q 533 426 547 435";
    }
    function _eyebrowTriangle(uint8 t) private pure returns (string memory) {
        if (t == 4) return "";
        if (t == 3) return "M 524 348 L 564 336";
        if (t == 2) return "M 524 350 L 562 342";
        if (t == 1) return "M 525 352 L 562 350";
        return             "M 526 352 L 558 360";
    }
    function _mouthSquare(uint8 t) private pure returns (string memory) {
        if (t == 4) return "M 519 313 Q 540 294 559 313";
        if (t == 3) return "M 523 314 Q 540 304 557 314";
        if (t == 2) return "M 524 315 Q 540 309 557 315";
        if (t == 1) return "M 525 315 L 556 315";
        return             "M 522 320 Q 540 326 558 318";
    }
    function _mouthHexagon(uint8 t) private pure returns (string memory) {
        if (t == 4) return "M 542 288 Q 562 322 582 288";
        if (t == 3) return "M 546 290 Q 562 310 578 290";
        if (t == 2) return "M 548 290 Q 562 300 576 290";
        if (t == 1) return "M 548 291 L 576 291";
        return             "M 546 298 Q 562 282 578 298";
    }

    // ─────────────────────────────────────────────────────────────
    //  Eye layer (includes blink animation)
    // ─────────────────────────────────────────────────────────────

    function _circleEye(uint8 creaTier, uint8 morTier) private pure returns (string memory) {
        string memory ry = _eyeRy(35, creaTier, morTier);
        string memory pr = _eyeScale(16, creaTier);
        string memory blinkRy;
        string memory blinkR;
        if (morTier == 4) {
            blinkRy = string(abi.encodePacked(' values="', ry, ';', ry, ';', ry, ';', ry, ';', ry, ';', ry, '" keyTimes="0;0.84;0.87;0.90;0.93;1" dur="7s" repeatCount="indefinite"'));
            blinkR  = string(abi.encodePacked(' values="', pr, ';', pr, ';', pr, ';', pr, ';', pr, ';', pr, '" keyTimes="0;0.84;0.87;0.90;0.93;1" dur="7s" repeatCount="indefinite"'));
        } else {
            blinkRy = string(abi.encodePacked(' values="', ry, ';', ry, ';1;1;', ry, ';', ry, '" keyTimes="0;0.84;0.87;0.90;0.93;1" dur="7s" repeatCount="indefinite"'));
            blinkR  = string(abi.encodePacked(' values="', pr, ';', pr, ';1;1;', pr, ';', pr, '" keyTimes="0;0.84;0.87;0.90;0.93;1" dur="7s" repeatCount="indefinite"'));
        }
        string memory blinkOp = ' values="1;1;0;0;1;1" keyTimes="0;0.84;0.87;0.90;0.93;1" dur="7s" repeatCount="indefinite"';
        return string(abi.encodePacked(
            '<ellipse cx="568" cy="252" rx="', _eyeScale(31, creaTier), '" ry="', ry, '" fill="white">',
            '<animate attributeName="ry"', blinkRy, '/></ellipse>',
            '<circle cx="', _circlePupilCx(morTier), '" cy="', _circlePupilCy(morTier),
            '" r="', pr, '" fill="#3b1200"><animate attributeName="r"', blinkR, '/></circle>',
            '<circle cx="572" cy="246" r="', _eyeScale(7, creaTier), '" fill="white">',
            '<animate attributeName="opacity"', blinkOp, '/></circle>'
        ));
    }

    function _triangleEye(uint8 creaTier, uint8 morTier) private pure returns (string memory) {
        return string(abi.encodePacked(
            '<ellipse cx="547" cy="370" rx="', _eyeScale(28, creaTier),
            '" ry="', _eyeRy(24, creaTier, morTier), '" fill="white"/>',
            '<circle cx="', _triPupilCx(morTier), '" cy="', _triPupilCy(morTier),
            '" r="', _eyeScale(13, creaTier), '" fill="#002918"/>',
            '<circle cx="553" cy="365" r="', _eyeScale(6, creaTier), '" fill="white"/>'
        ));
    }

    function _squareEye(uint8 creaTier, uint8 morTier) private pure returns (string memory) {
        return string(abi.encodePacked(
            '<ellipse cx="554" cy="244" rx="', _eyeScale(32, creaTier),
            '" ry="', _eyeRy(24, creaTier, morTier), '" fill="white"/>',
            '<circle cx="', _sqPupilCx(morTier), '" cy="', _sqPupilCy(morTier),
            '" r="', _eyeScale(13, creaTier), '" fill="#001533"/>',
            '<circle cx="558" cy="239" r="', _eyeScale(6, creaTier), '" fill="white"/>'
        ));
    }

    function _hexagonEye(uint8 creaTier, uint8 morTier, string memory lidColor) private pure returns (string memory) {
        return string(abi.encodePacked(
            '<ellipse cx="554" cy="232" rx="', _eyeScale(28, creaTier),
            '" ry="', _eyeRy(35, creaTier, morTier), '" fill="white"/>',
            '<path d="M 526 224 Q 554 208 582 224" fill="', lidColor, '"/>',
            '<circle cx="', _hexPupilCx(morTier), '" cy="', _hexPupilCy(morTier),
            '" r="', _eyeScale(15, creaTier), '" fill="#3b0020"/>',
            '<circle cx="562" cy="231" r="', _eyeScale(7, creaTier), '" fill="white"/>'
        ));
    }

    // ─────────────────────────────────────────────────────────────
    //  Shape layer (calls SoulShapes for body SVG / animation / gold ring)
    // ─────────────────────────────────────────────────────────────

    function _circleLayer(uint8 logicTier, uint8 decTier, uint8 creaTier, uint8 morTier, uint8 colorIdx, uint8 geomVar) private pure returns (string memory) {
        string memory gilded = logicTier == 4 ? SoulShapes.circleGildedRing(geomVar) : "";
        return string(abi.encodePacked(
            '<g transform="translate(0,80)">',
            SoulShapes.circleBody(geomVar, colorIdx, logicTier),
            gilded,
            _circleEye(creaTier, morTier),
            '<path d="', _mouthCircle(decTier), '" fill="none" stroke="#3b1200" stroke-width="4.5" stroke-linecap="round"/>',
            '</g>'
        ));
    }

    function _triangleLayer(uint8 logicTier, uint8 decTier, uint8 creaTier, uint8 morTier, uint8 colorIdx, uint8 geomVar) private pure returns (string memory) {
        string memory eyebrow   = _eyebrowTriangle(decTier);
        string memory gilded    = logicTier == 4 ? SoulShapes.triGildedRing(geomVar) : "";
        string memory eyebrowEl = bytes(eyebrow).length > 0
            ? string(abi.encodePacked('<path d="', eyebrow, '" stroke="#002918" stroke-width="4.5" stroke-linecap="round"/>'))
            : "";
        string memory part1 = string(abi.encodePacked(
            '<g transform="translate(0,80)">', SoulShapes.triShiverAnim(),
            SoulShapes.triBody(geomVar, colorIdx, logicTier),
            gilded, eyebrowEl
        ));
        return string(abi.encodePacked(
            part1,
            _triangleEye(creaTier, morTier),
            '<path d="', _mouthTriangle(decTier), '" fill="none" stroke="#002918" stroke-width="4.5" stroke-linecap="round"/>',
            '</g>'
        ));
    }

    function _squareLayer(uint8 logicTier, uint8 decTier, uint8 creaTier, uint8 morTier, uint8 colorIdx, uint8 geomVar) private pure returns (string memory) {
        string memory gilded = logicTier == 4 ? SoulShapes.sqGildedRing(geomVar) : "";
        string memory part1  = string(abi.encodePacked(
            '<g transform="translate(0,80)">',
            SoulShapes.sqAura(colorIdx, logicTier),
            '<g transform="rotate(', SoulShapes.sqBaseTilt(geomVar), ',400,285)"><g>',
            SoulShapes.sqTiltAnim(),
            SoulShapes.sqBody(geomVar, colorIdx)
        ));
        string memory part2  = string(abi.encodePacked(
            gilded,
            _squareEye(creaTier, morTier),
            '<path d="', _mouthSquare(decTier), '" fill="none" stroke="#001533" stroke-width="4" stroke-linecap="round"/>',
            '</g></g></g>'
        ));
        return string(abi.encodePacked(part1, part2));
    }

    function _hexagonLayer(uint8 logicTier, uint8 decTier, uint8 creaTier, uint8 morTier, uint8 colorIdx, uint8 geomVar) private pure returns (string memory) {
        string memory gilded = logicTier == 4 ? SoulShapes.hexGildedRing(geomVar) : "";
        string memory part1  = string(abi.encodePacked(
            '<g transform="translate(0,80)">', SoulShapes.hexHopAnim(),
            SoulShapes.hexBody(geomVar, colorIdx, logicTier),
            gilded
        ));
        return string(abi.encodePacked(
            part1,
            _hexagonEye(creaTier, morTier, SoulShapes.hexLid(colorIdx)),
            '<path d="', _mouthHexagon(decTier), '" fill="none" stroke="#3b0020" stroke-width="4.5" stroke-linecap="round"/>',
            '</g>'
        ));
    }

    function _getShapeLayer(
        bytes32 shapeHash,
        uint8 logicTier, uint8 decTier, uint8 creaTier, uint8 morTier,
        uint8 colorIdx, uint8 geomVar
    ) private pure returns (string memory) {
        if (shapeHash == keccak256("triangle")) return _triangleLayer(logicTier, decTier, creaTier, morTier, colorIdx, geomVar);
        if (shapeHash == keccak256("square"))   return _squareLayer(logicTier, decTier, creaTier, morTier, colorIdx, geomVar);
        if (shapeHash == keccak256("hexagon"))  return _hexagonLayer(logicTier, decTier, creaTier, morTier, colorIdx, geomVar);
        return _circleLayer(logicTier, decTier, creaTier, morTier, colorIdx, geomVar);
    }

    // ─────────────────────────────────────────────────────────────
    //  Public interface
    // ─────────────────────────────────────────────────────────────

    function renderSoulImage(
        string memory shape,
        uint256 logic,
        uint256 creativity,
        uint256 morality,
        uint256 decisiveness,
        uint256, /* chainID */
        uint256, /* unbindTimestamp */
        uint256 tokenId
    ) public pure returns (string memory) {
        bytes32 shapeHash = keccak256(abi.encodePacked(shape));
        uint8 rarity    = _rarityLevel(logic, creativity, morality, decisiveness);
        uint8 logicTier = _traitTier(logic);
        uint8 creaTier  = _traitTier(creativity);
        uint8 morTier   = _traitTier(morality);
        uint8 decTier   = _traitTier(decisiveness);
        uint8 colorIdx  = uint8(tokenId % 8);
        uint8 geomVar   = uint8((tokenId / 8) % 5);

        return string(abi.encodePacked(
            "data:image/svg+xml;base64,",
            Base64.encode(bytes(string(abi.encodePacked(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 800">',
                _getDefs(shapeHash, rarity, tokenId),
                _getBackground(rarity),
                _getShapeLayer(shapeHash, logicTier, decTier, creaTier, morTier, colorIdx, geomVar),
                '</svg>'
            ))))
        ));
    }
}


// ===== FILE: contracts/flatworld/SoulMint.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/interfaces/IERC2981.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Strings.sol";
import "@openzeppelin/contracts/utils/Base64.sol";
import "./FlatModel.sol";

/**
 * @title SoulMint
 * @notice 5,000 Flatworld Souls — direct public mint, 100% on-chain SVG.
 *
 * Supply:
 *   MAX_SUPPLY        5,000
 *   FREE_MINT_SUPPLY  3,000  (1 per wallet)
 *   PAID_MINT_SUPPLY  2,000  (0.003 ETH each, max 10 per tx)
 *
 * Shape distribution (total 5,000):
 *   circle    2,000
 *   square    1,500
 *   triangle  1,000
 *   hexagon     500
 *
 * Royalty: 5% ERC-2981, receiver = owner
 */
contract SoulMint is ERC721, IERC2981, ReentrancyGuard, Ownable {
    using Strings for uint256;

    // ─────────────────────────────────────────────────────────────
    //  Constants
    // ─────────────────────────────────────────────────────────────

    uint256 public constant MAX_SUPPLY       = 5_000;
    uint256 public constant FREE_MINT_SUPPLY = 3_000;
    uint256 public constant PAID_PRICE       = 0.003 ether;
    uint256 public constant MAX_PAID_PER_TX  = 10;
    uint96  public constant ROYALTY_BPS      = 500; // 5%

    // ─────────────────────────────────────────────────────────────
    //  Storage
    // ─────────────────────────────────────────────────────────────

    uint256 public totalMinted;
    uint256 public freeMinted;

    mapping(address => bool) public hasFreeMinted;

    FlatModel public immutable flatModel;

    struct Soul {
        uint256 logic;
        uint256 creativity;
        uint256 morality;
        uint256 decisiveness;
        uint8   shape; // 0=circle 1=square 2=triangle 3=hexagon
    }

    mapping(uint256 => Soul) public souls;

    // Shape quotas: circle 2000 / square 1500 / triangle 1000 / hexagon 500
    uint256[4] private _shapeRemaining;

    // ─────────────────────────────────────────────────────────────
    //  Events
    // ─────────────────────────────────────────────────────────────

    event FreeMinted(address indexed to, uint256 indexed tokenId);
    event PaidMinted(address indexed to, uint256 indexed firstTokenId, uint256 qty);

    // ─────────────────────────────────────────────────────────────
    //  Constructor
    // ─────────────────────────────────────────────────────────────

    constructor(address _flatModel) ERC721("Flatworld Soul", "FWS") {
        flatModel = FlatModel(_flatModel);
        _shapeRemaining[0] = 2_000; // circle
        _shapeRemaining[1] = 1_500; // square
        _shapeRemaining[2] = 1_000; // triangle
        _shapeRemaining[3] =   500; // hexagon
    }

    // ─────────────────────────────────────────────────────────────
    //  Mint
    // ─────────────────────────────────────────────────────────────

    /// @notice Claim 1 free Soul. One per wallet, first 3,000 only.
    function freeMint() external nonReentrant {
        require(!hasFreeMinted[msg.sender], "Already claimed");
        require(freeMinted < FREE_MINT_SUPPLY, "Free supply exhausted");
        require(totalMinted < MAX_SUPPLY, "Sold out");

        hasFreeMinted[msg.sender] = true;
        freeMinted++;
        uint256 tokenId = _mintSoul(msg.sender);
        emit FreeMinted(msg.sender, tokenId);
    }

    /// @notice Mint 1–10 Souls at 0.003 ETH each.
    function paidMint(uint256 qty) external payable nonReentrant {
        require(qty > 0 && qty <= MAX_PAID_PER_TX, "Quantity must be 1-10");
        require(msg.value == PAID_PRICE * qty, "Wrong ETH amount");
        require(totalMinted + qty <= MAX_SUPPLY, "Exceeds max supply");

        uint256 firstTokenId = totalMinted + 1;
        for (uint256 i = 0; i < qty; i++) {
            _mintSoul(msg.sender);
        }
        emit PaidMinted(msg.sender, firstTokenId, qty);
    }

    function _mintSoul(address to) internal returns (uint256 tokenId) {
        totalMinted++;
        tokenId = totalMinted;

        uint256 random = uint256(keccak256(abi.encodePacked(
            block.prevrandao,
            block.timestamp,
            to,
            tokenId
        )));

        uint8 shape = _selectShape(random);

        // Attributes distributed across 20-99 range; each attr uses a different slice
        uint256 logic        = 20 + (random >>  32) % 80;
        uint256 creativity   = 20 + (random >>  64) % 80;
        uint256 morality     = 20 + (random >>  96) % 80;
        uint256 decisiveness = 20 + (random >> 128) % 80;

        souls[tokenId] = Soul(logic, creativity, morality, decisiveness, shape);
        _safeMint(to, tokenId);
    }

    function _selectShape(uint256 random) internal returns (uint8) {
        uint256 total = _shapeRemaining[0] + _shapeRemaining[1]
                      + _shapeRemaining[2] + _shapeRemaining[3];
        if (total == 0) return uint8(random % 4); // fallback: all quotas zero
        uint256 roll = random % total;
        uint256 cumulative = 0;
        for (uint8 i = 0; i < 4; i++) {
            cumulative += _shapeRemaining[i];
            if (roll < cumulative) {
                _shapeRemaining[i]--;
                return i;
            }
        }
        return 0;
    }

    // ─────────────────────────────────────────────────────────────
    //  tokenURI — 100% on-chain SVG via FlatModel
    // ─────────────────────────────────────────────────────────────

    function tokenURI(uint256 tokenId) public view override returns (string memory) {
        require(_exists(tokenId), "Token does not exist");
        Soul storage s = souls[tokenId];

        string memory image = flatModel.renderSoulImage(
            _shapeName(s.shape),
            s.logic, s.creativity, s.morality, s.decisiveness,
            block.chainid,
            0,
            tokenId
        );

        bytes memory metadata = abi.encodePacked(
            '{"name":"Flatworld Soul #', tokenId.toString(),
            '","description":"A unique geometric Soul born in Flatworld. 100% on-chain, no IPFS, no servers.",',
            '"image":"', image, '",',
            '"attributes":', _buildAttributes(s),
            '}'
        );

        return string(abi.encodePacked(
            "data:application/json;base64,",
            Base64.encode(metadata)
        ));
    }

    // ─────────────────────────────────────────────────────────────
    //  ERC-2981 Royalty
    // ─────────────────────────────────────────────────────────────

    function royaltyInfo(uint256, uint256 salePrice)
        external
        view
        override
        returns (address receiver, uint256 royaltyAmount)
    {
        receiver = owner();
        royaltyAmount = (salePrice * ROYALTY_BPS) / 10_000;
    }

    // ─────────────────────────────────────────────────────────────
    //  Admin
    // ─────────────────────────────────────────────────────────────

    function withdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "Nothing to withdraw");
        (bool ok,) = payable(owner()).call{value: balance}("");
        require(ok, "Transfer failed");
    }

    // ─────────────────────────────────────────────────────────────
    //  supportsInterface
    // ─────────────────────────────────────────────────────────────

    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(ERC721, IERC165)
        returns (bool)
    {
        return interfaceId == type(IERC2981).interfaceId
            || super.supportsInterface(interfaceId);
    }

    // ─────────────────────────────────────────────────────────────
    //  Internal helpers
    // ─────────────────────────────────────────────────────────────

    function _shapeName(uint8 shape) internal pure returns (string memory) {
        if (shape == 0) return "circle";
        if (shape == 1) return "square";
        if (shape == 2) return "triangle";
        return "hexagon";
    }

    function _shapeDisplayName(uint8 shape) internal pure returns (string memory) {
        if (shape == 0) return "Circle";
        if (shape == 1) return "Square";
        if (shape == 2) return "Triangle";
        return "Hexagon";
    }

    function _rarityName(uint8 rarity) internal pure returns (string memory) {
        if (rarity == 4) return "LEGENDARY";
        if (rarity == 3) return "EPIC";
        if (rarity == 2) return "RARE";
        if (rarity == 1) return "UNCOMMON";
        return "COMMON";
    }

    function _rarityLevel(uint256 l, uint256 c, uint256 m, uint256 d) internal pure returns (uint8) {
        uint256 avg = (l + c + m + d) / 4;
        if (avg >= 90) return 4;
        if (avg >= 75) return 3;
        if (avg >= 60) return 2;
        if (avg >= 40) return 1;
        return 0;
    }

    function _buildAttributes(Soul storage s) internal view returns (string memory) {
        uint8 rarity = _rarityLevel(s.logic, s.creativity, s.morality, s.decisiveness);
        return string(abi.encodePacked(
            '[',
            '{"trait_type":"Shape","value":"',   _shapeDisplayName(s.shape), '"},',
            '{"trait_type":"Rarity","value":"',  _rarityName(rarity), '"},',
            '{"trait_type":"Glow","value":"',    flatModel.getGlowName(s.logic, s.creativity, s.morality, s.decisiveness), '"},',
            '{"trait_type":"Skin","value":"',    flatModel.getSkinName(s.logic, s.creativity, s.morality, s.decisiveness), '"},',
            '{"trait_type":"Mark","value":"',    flatModel.getMarkName(s.logic, s.creativity, s.morality, s.decisiveness), '"},',
            '{"trait_type":"Face","value":"',    flatModel.getFaceName(s.logic, s.creativity, s.morality, s.decisiveness), '"}',
            ']'
        ));
    }
}


// ===== FILE: contracts/flatworld/SoulShapes.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title SoulShapes
 * @notice External library: stores color variants, geometry variants, and shape body SVG builders.
 *         Called by FlatModel; extracted to a separate contract to stay within the 24 KB size limit.
 *
 * Color variants (colorIdx = tokenId % 8, 8 palettes per shape):
 *   circle   orange/amber/terracotta/coral (warm)  | triangle emerald/mint/lime/teal (green)
 *   square   cobalt/royal-blue/sky/indigo/violet/cyan | hexagon hot-pink/rose/magenta/violet (pink-purple)
 *
 * Geometry variants (geomVar = (tokenId / 8) % 5, 5 shapes per type):
 *   circle   standard/wide-ellipse/tall-ellipse/small/large
 *   triangle standard/tall-wide/short-wide/slightly-tall/asymmetric
 *   square   standard/wide-flat/tall/rounded/wide+extra-rounded
 *   hexagon  1.00x/0.93x/1.06x/0.97x/1.03x scale
 */
library SoulShapes {

    // ─────────────────────────────────────────────────────────────
    //  Private helper: aura opacity
    // ─────────────────────────────────────────────────────────────

    function _auraOpacity(uint8 tier) private pure returns (string memory) {
        if (tier == 4) return "0.65";
        if (tier == 3) return "0.45";
        if (tier == 2) return "0.28";
        if (tier == 1) return "0.15";
        return "0.06";
    }

    // ─────────────────────────────────────────────────────────────
    //  Private helper: per-shape color lookup (fill + stroke + aura)
    // ─────────────────────────────────────────────────────────────

    function _circleColors(uint8 i) private pure returns (
        string memory fill, string memory stroke, string memory aura
    ) {
        if (i == 1) return ("rgba(230,59,0,0.92)",   "#ff7033", "#cc3300");
        if (i == 2) return ("rgba(255,172,0,0.92)",  "#ffd055", "#cc8800");
        if (i == 3) return ("rgba(196,82,0,0.92)",   "#e87533", "#993300");
        if (i == 4) return ("rgba(255,112,85,0.92)", "#ffaa99", "#cc5544");
        if (i == 5) return ("rgba(221,34,0,0.92)",   "#ff6644", "#aa1100");
        if (i == 6) return ("rgba(255,153,0,0.92)",  "#ffcc44", "#cc8800");
        if (i == 7) return ("rgba(204,85,51,0.92)",  "#ee8866", "#aa4422");
        return             ("rgba(248,130,42,0.92)", "#ffaa44", "#ff8c00");
    }

    function _triColors(uint8 i) private pure returns (
        string memory fill, string memory stroke, string memory aura
    ) {
        // 8 variants: dark-forest → emerald → lime → spring-green → cyan-blue → deep-jade → yellow-green → teal
        if (i == 1) return ("rgba(21,128,61,0.92)",   "#22c55e", "#14532d");  // dark forest green
        if (i == 2) return ("rgba(132,204,22,0.92)",  "#bef264", "#4d7c0f");  // lime yellow-green
        if (i == 3) return ("rgba(6,182,212,0.92)",   "#67e8f9", "#0e7490");  // cyan blue
        if (i == 4) return ("rgba(74,222,128,0.92)",  "#86efac", "#166534");  // bright spring green
        if (i == 5) return ("rgba(5,120,87,0.92)",    "#34d399", "#064e3b");  // deep jade green
        if (i == 6) return ("rgba(163,230,53,0.92)",  "#d9f99d", "#365314");  // yellow green
        if (i == 7) return ("rgba(20,184,166,0.92)",  "#5eead4", "#115e59");  // teal
        return             ("rgba(16,185,129,0.92)",  "#34d399", "#059669");  // emerald (default)
    }

    function _sqColors(uint8 i) private pure returns (
        string memory fill, string memory stroke, string memory aura
    ) {
        if (i == 1) return ("rgba(29,78,216,0.92)",   "#3b82f6", "#1e40af");
        if (i == 2) return ("rgba(56,189,248,0.92)",  "#7dd3fc", "#0284c7");
        if (i == 3) return ("rgba(99,102,241,0.92)",  "#a5b4fc", "#4338ca");
        if (i == 4) return ("rgba(124,58,237,0.92)",  "#a78bfa", "#6d28d9");
        if (i == 5) return ("rgba(8,145,178,0.92)",   "#22d3ee", "#0e7490");
        if (i == 6) return ("rgba(79,159,255,0.92)",  "#93c5fd", "#2563eb");
        if (i == 7) return ("rgba(14,165,233,0.92)",  "#7dd3fc", "#0369a1");
        return             ("rgba(59,130,246,0.92)",  "#60a5fa", "#1d4ed8");
    }

    function _hexColors(uint8 i) private pure returns (
        string memory fill, string memory stroke, string memory aura
    ) {
        if (i == 1) return ("rgba(190,24,93,0.92)",   "#ec4899", "#9d174d");
        if (i == 2) return ("rgba(244,114,182,0.92)", "#fbcfe8", "#be185d");
        if (i == 3) return ("rgba(217,70,239,0.92)",  "#f0abfc", "#a21caf");
        if (i == 4) return ("rgba(168,85,247,0.92)",  "#d8b4fe", "#7e22ce");
        if (i == 5) return ("rgba(219,39,119,0.92)",  "#f472b6", "#9d174d");
        if (i == 6) return ("rgba(255,112,196,0.92)", "#ffc0e0", "#db2777");
        if (i == 7) return ("rgba(192,38,211,0.92)",  "#e879b9", "#86198f");
        return             ("rgba(236,72,153,0.92)",  "#f472b6", "#db2777");
    }

    // ─────────────────────────────────────────────────────────────
    //  Private helper: geometry variant lookup
    // ─────────────────────────────────────────────────────────────

    function _sqShadowRect(uint8 v) private pure returns (string memory) {
        if (v == 1) return '<rect x="212" y="132" width="376" height="306" rx="4"';
        if (v == 2) return '<rect x="247" y="97"  width="306" height="376" rx="4"';
        if (v == 3) return '<rect x="232" y="117" width="336" height="336" rx="44"';
        if (v == 4) return '<rect x="217" y="132" width="366" height="306" rx="58"';
        return              '<rect x="232" y="117" width="336" height="336" rx="4"';
    }

    function _sqBodyRect(uint8 v) private pure returns (string memory) {
        if (v == 1) return '<rect x="220" y="140" width="360" height="290" rx="4"';
        if (v == 2) return '<rect x="255" y="105" width="290" height="360" rx="4"';
        if (v == 3) return '<rect x="240" y="125" width="320" height="320" rx="36"';
        if (v == 4) return '<rect x="225" y="140" width="350" height="290" rx="50"';
        return              '<rect x="240" y="125" width="320" height="320" rx="4"';
    }

    function _triPoints(uint8 v) private pure returns (string memory) {
        // v=0 standard | v=1 tall+wide-base | v=2 short+wide-base
        // v=3 mid-tall+slightly-wide | v=4 standard-tall+wide-base
        // Note: all variants ensure the right edge at y=370 is >= x558 so eyes stay in bounds
        if (v == 1) return "400,55 145,490 655,490";   // tall+wide-base: apex +35px higher, base +60px wider
        if (v == 2) return "400,135 165,475 635,475";  // short+wide-base: apex -45px lower, base +30px wider
        if (v == 3) return "400,75 160,485 640,485";   // mid-tall+slightly-wide
        if (v == 4) return "400,100 155,480 645,480";  // standard-tall+wide-base
        return              "400,90 175,478 625,478";   // standard
    }

    function _hexPoints(uint8 v) private pure returns (string memory) {
        if (v == 1) return "400,104 557,195 557,376 400,466 243,376 243,195";
        if (v == 2) return "400,78 579,182 579,390 400,492 221,390 221,182";
        if (v == 3) return "400,96 564,191 564,380 400,474 236,380 236,191";
        if (v == 4) return "400,84 574,185 574,386 400,486 226,386 226,185";
        return              "400,90 569,188 569,383 400,480 231,383 231,188";
    }

    // ─────────────────────────────────────────────────────────────
    //  Public interface: shape body SVG (aura + body + texture overlay)
    // ─────────────────────────────────────────────────────────────

    function circleBody(uint8 geomVar, uint8 colorIdx, uint8 logicTier) public pure returns (string memory) {
        (string memory fill, string memory stroke, string memory aura) = _circleColors(colorIdx);
        string memory auraEl = string(abi.encodePacked(
            '<circle cx="400" cy="285" r="210" fill="', aura,
            '" opacity="', _auraOpacity(logicTier), '" filter="url(#halo)"/>'
        ));
        string memory body;
        string memory tex;
        if (geomVar == 1) {
            body = string(abi.encodePacked('<ellipse cx="400" cy="285" rx="208" ry="170" fill="', fill, '" stroke="', stroke, '" stroke-width="4"/>'));
            tex  = '<ellipse cx="400" cy="285" rx="208" ry="170" fill="rgba(255,235,160,0.55)" filter="url(#tex)"/>';
        } else if (geomVar == 2) {
            body = string(abi.encodePacked('<ellipse cx="400" cy="285" rx="170" ry="208" fill="', fill, '" stroke="', stroke, '" stroke-width="4"/>'));
            tex  = '<ellipse cx="400" cy="285" rx="170" ry="208" fill="rgba(255,235,160,0.55)" filter="url(#tex)"/>';
        } else if (geomVar == 3) {
            body = string(abi.encodePacked('<circle cx="400" cy="285" r="176" fill="', fill, '" stroke="', stroke, '" stroke-width="4"/>'));
            tex  = '<circle cx="400" cy="285" r="176" fill="rgba(255,235,160,0.55)" filter="url(#tex)"/>';
        } else if (geomVar == 4) {
            body = string(abi.encodePacked('<circle cx="400" cy="285" r="200" fill="', fill, '" stroke="', stroke, '" stroke-width="4"/>'));
            tex  = '<circle cx="400" cy="285" r="200" fill="rgba(255,235,160,0.55)" filter="url(#tex)"/>';
        } else {
            body = string(abi.encodePacked('<circle cx="400" cy="285" r="190" fill="', fill, '" stroke="', stroke, '" stroke-width="4"/>'));
            tex  = '<circle cx="400" cy="285" r="190" fill="rgba(255,235,160,0.55)" filter="url(#tex)"/>';
        }
        return string(abi.encodePacked(auraEl, body, tex));
    }

    function triBody(uint8 geomVar, uint8 colorIdx, uint8 logicTier) public pure returns (string memory) {
        (string memory fill, string memory stroke, string memory aura) = _triColors(colorIdx);
        string memory pts = _triPoints(geomVar);
        return string(abi.encodePacked(
            '<polygon points="', pts, '" fill="', aura, '" opacity="', _auraOpacity(logicTier), '" filter="url(#halo)"/>',
            '<polygon points="', pts, '" fill="', fill, '" stroke="', stroke, '" stroke-width="4" stroke-linejoin="round"/>',
            '<polygon points="', pts, '" fill="rgba(0,0,0,0.55)" filter="url(#tex)"/>'
        ));
    }

    function sqBody(uint8 geomVar, uint8 colorIdx) public pure returns (string memory) {
        (string memory fill, string memory stroke,) = _sqColors(colorIdx);
        string memory sr = _sqShadowRect(geomVar);
        string memory br = _sqBodyRect(geomVar);
        return string(abi.encodePacked(
            sr, ' fill="', fill, '" opacity="0.25" filter="url(#halo)"/>',
            br, ' fill="', fill, '" stroke="', stroke, '" stroke-width="4"/>',
            br, ' fill="rgba(0,0,0,0.55)" filter="url(#tex)"/>'
        ));
    }

    function sqAura(uint8 colorIdx, uint8 logicTier) public pure returns (string memory) {
        (,, string memory aura) = _sqColors(colorIdx);
        return string(abi.encodePacked(
            '<circle cx="400" cy="285" r="210" fill="', aura,
            '" opacity="', _auraOpacity(logicTier), '" filter="url(#halo)"/>'
        ));
    }

    function hexBody(uint8 geomVar, uint8 colorIdx, uint8 logicTier) public pure returns (string memory) {
        (string memory fill, string memory stroke, string memory aura) = _hexColors(colorIdx);
        string memory pts = _hexPoints(geomVar);
        return string(abi.encodePacked(
            '<polygon points="', pts, '" fill="', aura, '" opacity="', _auraOpacity(logicTier), '" filter="url(#halo)"/>',
            '<polygon points="', pts, '" fill="', fill, '" stroke="', stroke, '" stroke-width="4" stroke-linejoin="round"/>',
            '<polygon points="', pts, '" fill="rgba(0,0,0,0.55)" filter="url(#tex)"/>'
        ));
    }

    // Hexagon upper eyelid color (follows primary color)
    function hexLid(uint8 colorIdx) public pure returns (string memory) {
        if (colorIdx == 1) return "#be185d";
        if (colorIdx == 2) return "#f472b6";
        if (colorIdx == 3) return "#d946ef";
        if (colorIdx == 4) return "#a855f7";
        if (colorIdx == 5) return "#db2777";
        if (colorIdx == 6) return "#ff70c4";
        if (colorIdx == 7) return "#c026d3";
        return "#ec4899";
    }

    // ─────────────────────────────────────────────────────────────
    //  Public interface: geometry parameters (used by FlatModel layer functions)
    // ─────────────────────────────────────────────────────────────

    function sqBaseTilt(uint8 geomVar) public pure returns (string memory) {
        if (geomVar == 1) return "-5";
        if (geomVar == 2) return "-9";
        if (geomVar == 4) return "-6";
        return "-7";
    }

    // ─────────────────────────────────────────────────────────────
    //  Public interface: GILDED gold ring (LEGENDARY exclusive)
    // ─────────────────────────────────────────────────────────────

    function circleGildedRing(uint8 geomVar) public pure returns (string memory) {
        string memory r;
        if      (geomVar == 3) r = "181";
        else if (geomVar == 4) r = "205";
        else                   r = "195";
        return string(abi.encodePacked(
            '<circle cx="400" cy="285" r="', r, '" fill="none" stroke="#FFD700" stroke-width="3" opacity="0.45" filter="url(#halo)"/>'
        ));
    }

    function triGildedRing(uint8 geomVar) public pure returns (string memory) {
        return string(abi.encodePacked(
            '<polygon points="', _triPoints(geomVar),
            '" fill="none" stroke="#FFD700" stroke-width="3" opacity="0.45" filter="url(#halo)"/>'
        ));
    }

    function sqGildedRing(uint8 geomVar) public pure returns (string memory) {
        if (geomVar == 1) return '<rect x="218" y="138" width="364" height="294" rx="4" fill="none" stroke="#FFD700" stroke-width="3" opacity="0.45" filter="url(#halo)"/>';
        if (geomVar == 2) return '<rect x="253" y="103" width="294" height="364" rx="4" fill="none" stroke="#FFD700" stroke-width="3" opacity="0.45" filter="url(#halo)"/>';
        if (geomVar == 3) return '<rect x="238" y="123" width="324" height="324" rx="38" fill="none" stroke="#FFD700" stroke-width="3" opacity="0.45" filter="url(#halo)"/>';
        if (geomVar == 4) return '<rect x="223" y="138" width="354" height="294" rx="52" fill="none" stroke="#FFD700" stroke-width="3" opacity="0.45" filter="url(#halo)"/>';
        return              '<rect x="238" y="123" width="324" height="324" rx="4" fill="none" stroke="#FFD700" stroke-width="3" opacity="0.45" filter="url(#halo)"/>';
    }

    function hexGildedRing(uint8 geomVar) public pure returns (string memory) {
        return string(abi.encodePacked(
            '<polygon points="', _hexPoints(geomVar), '" fill="none" stroke="#FFD700" stroke-width="3" opacity="0.45" filter="url(#halo)"/>'
        ));
    }

    // ─────────────────────────────────────────────────────────────
    //  Public interface: animation strings
    // ─────────────────────────────────────────────────────────────

    function sqTiltAnim() public pure returns (string memory) {
        return string(abi.encodePacked(
            '<animateTransform attributeName="transform" type="rotate"',
            ' values="0 400 285;0 400 285;-4 400 285;0 400 285;0 400 285"',
            ' keyTimes="0;0.78;0.83;0.88;1"',
            ' dur="9s" repeatCount="indefinite" calcMode="spline"',
            ' keySplines="0 0 1 1;0.2 0 0.8 1;0.8 0 1 0;0 0 1 1"/>'
        ));
    }

    function triShiverAnim() public pure returns (string memory) {
        return string(abi.encodePacked(
            '<animateTransform attributeName="transform" type="translate" additive="sum"',
            ' values="0 0;0 0;-4 -10;4 -8;0 0;0 0"',
            ' keyTimes="0;0.720;0.755;0.780;0.810;1"',
            ' dur="9s" repeatCount="indefinite" calcMode="spline"',
            ' keySplines="0 0 1 1;0.3 0 0.7 1;0.3 0 0.7 1;0.3 0 0.7 1;0 0 1 1"/>'
        ));
    }

    function hexHopAnim() public pure returns (string memory) {
        return string(abi.encodePacked(
            '<animateTransform attributeName="transform" type="translate" additive="sum"',
            ' values="0 0;0 0;0 -14;0 -4;0 0;0 0"',
            ' keyTimes="0;0.800;0.835;0.865;0.895;1"',
            ' dur="8s" repeatCount="indefinite" calcMode="spline"',
            ' keySplines="0 0 1 1;0.2 0 0.8 1;0.6 0 1 0.4;0.4 0 0.6 1;0 0 1 1"/>'
        ));
    }
}
