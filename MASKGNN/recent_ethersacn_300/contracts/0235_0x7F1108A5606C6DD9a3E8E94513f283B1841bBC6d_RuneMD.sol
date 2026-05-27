// ===== FILE: src/RuneMD.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ERC721} from "solady/tokens/ERC721.sol";
import {ERC2981} from "solady/tokens/ERC2981.sol";
import {Ownable} from "solady/auth/Ownable.sol";
import {LibString} from "solady/utils/LibString.sol";
import {Base64} from "solady/utils/Base64.sol";
import {RuneRenderer} from "./RuneRenderer.sol";

/// @title rune.md — 5555 fully on-chain procedural runes
/// @notice First 1000 free, remaining 4555 at 0.00022 ETH. Max 10 per wallet.
///         ERC2981 royalty is redirected to a random current holder on every secondary sale.
contract RuneMD is ERC721, ERC2981, Ownable {
    using LibString for uint256;

    uint256 public constant MAX_SUPPLY = 5555;
    uint256 public constant FREE_SUPPLY = 1000;
    uint256 public constant PAID_PRICE = 0.00022 ether;
    uint256 public constant MAX_PER_WALLET = 10;
    uint96  public constant ROYALTY_BPS = 500; // 5%

    RuneRenderer public immutable renderer;

    uint256 public totalSupply;
    uint256 public mintRevenue; // owner-withdrawable
    mapping(address minter => uint256 count) public mintedBy;

    event Minted(address indexed minter, uint256 indexed firstId, uint256 amount, uint256 paid);
    event RoyaltyForwarded(uint256 indexed toId, address indexed holder, uint256 amount);
    event RoyaltyOrphaned(uint256 amount); // received with no holders

    error MintPriceWrong();
    error WalletLimit();
    error SupplyExceeded();
    error ZeroAmount();
    error NothingToWithdraw();

    constructor(RuneRenderer renderer_, address owner_) {
        renderer = renderer_;
        _initializeOwner(owner_);
        _setDefaultRoyalty(address(this), ROYALTY_BPS); // route royalty to this contract
    }

    function name() public pure override returns (string memory) { return "rune.md"; }
    function symbol() public pure override returns (string memory) { return "RUNE"; }

    /// @notice cost of minting `amount` if it started at `from` (totalSupply at the time).
    /// @dev pure — useful for the frontend to display dynamic cost.
    function priceFor(uint256 from, uint256 amount) public pure returns (uint256) {
        if (amount == 0) return 0;
        uint256 to = from + amount;
        if (to <= FREE_SUPPLY) return 0;
        if (from >= FREE_SUPPLY) return amount * PAID_PRICE;
        return (to - FREE_SUPPLY) * PAID_PRICE;
    }

    /// @notice convenience: how much to send for `amount` minted right now.
    function currentPriceFor(uint256 amount) external view returns (uint256) {
        return priceFor(totalSupply, amount);
    }

    function mint(uint256 amount) external payable {
        if (amount == 0) revert ZeroAmount();
        if (mintedBy[msg.sender] + amount > MAX_PER_WALLET) revert WalletLimit();
        uint256 first = totalSupply;
        uint256 next = first + amount;
        if (next > MAX_SUPPLY) revert SupplyExceeded();

        uint256 cost = priceFor(first, amount);
        if (msg.value != cost) revert MintPriceWrong();

        mintedBy[msg.sender] += amount;
        totalSupply = next;
        if (cost > 0) mintRevenue += cost;

        for (uint256 i = first; i < next; ++i) _mint(msg.sender, i);
        emit Minted(msg.sender, first, amount, cost);
    }

    function tokenURI(uint256 id) public view override returns (string memory) {
        ownerOf(id); // existence check (reverts if unminted/burned)
        string memory image = renderer.svg(id);
        string memory json = string.concat(
            '{"name":"rune ', _pad4(id),
            '","description":"on-chain procedural rune. 5555 supply. each glyph unique.",',
            '"image":"data:image/svg+xml;base64,', Base64.encode(bytes(image)), '"}'
        );
        return string.concat("data:application/json;base64,", Base64.encode(bytes(json)));
    }

    /// @dev royalty receipts (or arbitrary ETH) flow to one random current holder.
    receive() external payable {
        if (msg.value == 0) return;
        if (totalSupply == 0) {
            emit RoyaltyOrphaned(msg.value);
            return;
        }
        uint256 r = uint256(
            keccak256(abi.encodePacked(block.prevrandao, block.timestamp, msg.sender, msg.value, totalSupply))
        ) % totalSupply;
        address holder = ownerOf(r);
        (bool ok,) = holder.call{value: msg.value, gas: 30_000}("");
        if (!ok) {
            emit RoyaltyOrphaned(msg.value); // funds remain in contract; owner can sweep via withdraw
        } else {
            emit RoyaltyForwarded(r, holder, msg.value);
        }
    }

    function withdraw() external onlyOwner {
        uint256 amt = mintRevenue;
        if (amt == 0) revert NothingToWithdraw();
        mintRevenue = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }

    /// @notice sweep orphaned royalty payments (failed forwards) — never touches mintRevenue.
    function sweepOrphaned() external onlyOwner {
        uint256 amt = address(this).balance - mintRevenue;
        if (amt == 0) revert NothingToWithdraw();
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }

    // ── interfaces ──────────────────────────────────────────
    /// @dev explicit list so block explorers can detect token type without ambiguity.
    function supportsInterface(bytes4 id) public pure override(ERC721, ERC2981) returns (bool) {
        return id == 0x01ffc9a7  // ERC165
            || id == 0x80ac58cd  // ERC721
            || id == 0x5b5e139f  // ERC721Metadata
            || id == 0x2a55205a; // ERC2981
    }

    function _pad4(uint256 n) internal pure returns (string memory) {
        string memory s = n.toString();
        if (n < 10) return string.concat("000", s);
        if (n < 100) return string.concat("00", s);
        if (n < 1000) return string.concat("0", s);
        return s;
    }
}


// ===== FILE: lib/solady/src/tokens/ERC721.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

/// @notice Simple ERC721 implementation with storage hitchhiking.
/// @author Solady (https://github.com/vectorized/solady/blob/main/src/tokens/ERC721.sol)
/// @author Modified from Solmate (https://github.com/transmissions11/solmate/blob/main/src/tokens/ERC721.sol)
/// @author Modified from OpenZeppelin (https://github.com/OpenZeppelin/openzeppelin-contracts/tree/master/contracts/token/ERC721/ERC721.sol)
///
/// @dev Note:
/// - The ERC721 standard allows for self-approvals.
///   For performance, this implementation WILL NOT revert for such actions.
///   Please add any checks with overrides if desired.
/// - For performance, methods are made payable where permitted by the ERC721 standard.
/// - The `safeTransfer` functions use the identity precompile (0x4)
///   to copy memory internally.
///
/// If you are overriding:
/// - NEVER violate the ERC721 invariant:
///   the balance of an owner MUST always be equal to their number of ownership slots.
///   The transfer functions do not have an underflow guard for user token balances.
/// - Make sure all variables written to storage are properly cleaned
///   (e.g. the bool value for `isApprovedForAll` MUST be either 1 or 0 under the hood).
/// - Check that the overridden function is actually used in the function you want to
///   change the behavior of. Much of the code has been manually inlined for performance.
abstract contract ERC721 {
    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                         CONSTANTS                          */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev An account can hold up to 4294967295 tokens.
    uint256 internal constant _MAX_ACCOUNT_BALANCE = 0xffffffff;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                       CUSTOM ERRORS                        */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Only the token owner or an approved account can manage the token.
    error NotOwnerNorApproved();

    /// @dev The token does not exist.
    error TokenDoesNotExist();

    /// @dev The token already exists.
    error TokenAlreadyExists();

    /// @dev Cannot query the balance for the zero address.
    error BalanceQueryForZeroAddress();

    /// @dev Cannot mint or transfer to the zero address.
    error TransferToZeroAddress();

    /// @dev The token must be owned by `from`.
    error TransferFromIncorrectOwner();

    /// @dev The recipient's balance has overflowed.
    error AccountBalanceOverflow();

    /// @dev Cannot safely transfer to a contract that does not implement
    /// the ERC721Receiver interface.
    error TransferToNonERC721ReceiverImplementer();

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                           EVENTS                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Emitted when token `id` is transferred from `from` to `to`.
    event Transfer(address indexed from, address indexed to, uint256 indexed id);

    /// @dev Emitted when `owner` enables `account` to manage the `id` token.
    event Approval(address indexed owner, address indexed account, uint256 indexed id);

    /// @dev Emitted when `owner` enables or disables `operator` to manage all of their tokens.
    event ApprovalForAll(address indexed owner, address indexed operator, bool isApproved);

    /// @dev `keccak256(bytes("Transfer(address,address,uint256)"))`.
    uint256 private constant _TRANSFER_EVENT_SIGNATURE =
        0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef;

    /// @dev `keccak256(bytes("Approval(address,address,uint256)"))`.
    uint256 private constant _APPROVAL_EVENT_SIGNATURE =
        0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925;

    /// @dev `keccak256(bytes("ApprovalForAll(address,address,bool)"))`.
    uint256 private constant _APPROVAL_FOR_ALL_EVENT_SIGNATURE =
        0x17307eab39ab6107e8899845ad3d59bd9653f200f220920489ca2b5937696c31;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                          STORAGE                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev The ownership data slot of `id` is given by:
    /// ```
    ///     mstore(0x00, id)
    ///     mstore(0x1c, _ERC721_MASTER_SLOT_SEED)
    ///     let ownershipSlot := add(id, add(id, keccak256(0x00, 0x20)))
    /// ```
    /// Bits Layout:
    /// - [0..159]   `addr`
    /// - [160..255] `extraData`
    ///
    /// The approved address slot is given by: `add(1, ownershipSlot)`.
    ///
    /// See: https://notes.ethereum.org/%40vbuterin/verkle_tree_eip
    ///
    /// The balance slot of `owner` is given by:
    /// ```
    ///     mstore(0x1c, _ERC721_MASTER_SLOT_SEED)
    ///     mstore(0x00, owner)
    ///     let balanceSlot := keccak256(0x0c, 0x1c)
    /// ```
    /// Bits Layout:
    /// - [0..31]   `balance`
    /// - [32..255] `aux`
    ///
    /// The `operator` approval slot of `owner` is given by:
    /// ```
    ///     mstore(0x1c, or(_ERC721_MASTER_SLOT_SEED, operator))
    ///     mstore(0x00, owner)
    ///     let operatorApprovalSlot := keccak256(0x0c, 0x30)
    /// ```
    uint256 private constant _ERC721_MASTER_SLOT_SEED = 0x7d8825530a5a2e7a << 192;

    /// @dev Pre-shifted and pre-masked constant.
    uint256 private constant _ERC721_MASTER_SLOT_SEED_MASKED = 0x0a5a2e7a00000000;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                      ERC721 METADATA                       */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Returns the token collection name.
    function name() public view virtual returns (string memory);

    /// @dev Returns the token collection symbol.
    function symbol() public view virtual returns (string memory);

    /// @dev Returns the Uniform Resource Identifier (URI) for token `id`.
    function tokenURI(uint256 id) public view virtual returns (string memory);

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                           ERC721                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Returns the owner of token `id`.
    ///
    /// Requirements:
    /// - Token `id` must exist.
    function ownerOf(uint256 id) public view virtual returns (address result) {
        result = _ownerOf(id);
        /// @solidity memory-safe-assembly
        assembly {
            if iszero(result) {
                mstore(0x00, 0xceea21b6) // `TokenDoesNotExist()`.
                revert(0x1c, 0x04)
            }
        }
    }

    /// @dev Returns the number of tokens owned by `owner`.
    ///
    /// Requirements:
    /// - `owner` must not be the zero address.
    function balanceOf(address owner) public view virtual returns (uint256 result) {
        /// @solidity memory-safe-assembly
        assembly {
            // Revert if the `owner` is the zero address.
            if iszero(owner) {
                mstore(0x00, 0x8f4eb604) // `BalanceQueryForZeroAddress()`.
                revert(0x1c, 0x04)
            }
            mstore(0x1c, _ERC721_MASTER_SLOT_SEED)
            mstore(0x00, owner)
            result := and(sload(keccak256(0x0c, 0x1c)), _MAX_ACCOUNT_BALANCE)
        }
    }

    /// @dev Returns the account approved to manage token `id`.
    ///
    /// Requirements:
    /// - Token `id` must exist.
    function getApproved(uint256 id) public view virtual returns (address result) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, id)
            mstore(0x1c, _ERC721_MASTER_SLOT_SEED)
            let ownershipSlot := add(id, add(id, keccak256(0x00, 0x20)))
            if iszero(shl(96, sload(ownershipSlot))) {
                mstore(0x00, 0xceea21b6) // `TokenDoesNotExist()`.
                revert(0x1c, 0x04)
            }
            result := sload(add(1, ownershipSlot))
        }
    }

    /// @dev Sets `account` as the approved account to manage token `id`.
    ///
    /// Requirements:
    /// - Token `id` must exist.
    /// - The caller must be the owner of the token,
    ///   or an approved operator for the token owner.
    ///
    /// Emits an {Approval} event.
    function approve(address account, uint256 id) public payable virtual {
        _approve(msg.sender, account, id);
    }

    /// @dev Returns whether `operator` is approved to manage the tokens of `owner`.
    function isApprovedForAll(address owner, address operator)
        public
        view
        virtual
        returns (bool result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x1c, operator)
            mstore(0x08, _ERC721_MASTER_SLOT_SEED_MASKED)
            mstore(0x00, owner)
            result := sload(keccak256(0x0c, 0x30))
        }
    }

    /// @dev Sets whether `operator` is approved to manage the tokens of the caller.
    ///
    /// Emits an {ApprovalForAll} event.
    function setApprovalForAll(address operator, bool isApproved) public virtual {
        /// @solidity memory-safe-assembly
        assembly {
            // Convert to 0 or 1.
            isApproved := iszero(iszero(isApproved))
            // Update the `isApproved` for (`msg.sender`, `operator`).
            mstore(0x1c, operator)
            mstore(0x08, _ERC721_MASTER_SLOT_SEED_MASKED)
            mstore(0x00, caller())
            sstore(keccak256(0x0c, 0x30), isApproved)
            // Emit the {ApprovalForAll} event.
            mstore(0x00, isApproved)
            // forgefmt: disable-next-item
            log3(0x00, 0x20, _APPROVAL_FOR_ALL_EVENT_SIGNATURE, caller(), shr(96, shl(96, operator)))
        }
    }

    /// @dev Transfers token `id` from `from` to `to`.
    ///
    /// Requirements:
    ///
    /// - Token `id` must exist.
    /// - `from` must be the owner of the token.
    /// - `to` cannot be the zero address.
    /// - The caller must be the owner of the token, or be approved to manage the token.
    ///
    /// Emits a {Transfer} event.
    function transferFrom(address from, address to, uint256 id) public payable virtual {
        _beforeTokenTransfer(from, to, id);
        /// @solidity memory-safe-assembly
        assembly {
            // Clear the upper 96 bits.
            let bitmaskAddress := shr(96, not(0))
            from := and(bitmaskAddress, from)
            to := and(bitmaskAddress, to)
            // Load the ownership data.
            mstore(0x00, id)
            mstore(0x1c, or(_ERC721_MASTER_SLOT_SEED, caller()))
            let ownershipSlot := add(id, add(id, keccak256(0x00, 0x20)))
            let ownershipPacked := sload(ownershipSlot)
            let owner := and(bitmaskAddress, ownershipPacked)
            // Revert if the token does not exist, or if `from` is not the owner.
            if iszero(mul(owner, eq(owner, from))) {
                // `TokenDoesNotExist()`, `TransferFromIncorrectOwner()`.
                mstore(shl(2, iszero(owner)), 0xceea21b6a1148100)
                revert(0x1c, 0x04)
            }
            // Load, check, and update the token approval.
            {
                mstore(0x00, from)
                let approvedAddress := sload(add(1, ownershipSlot))
                // Revert if the caller is not the owner, nor approved.
                if iszero(or(eq(caller(), from), eq(caller(), approvedAddress))) {
                    if iszero(sload(keccak256(0x0c, 0x30))) {
                        mstore(0x00, 0x4b6e7f18) // `NotOwnerNorApproved()`.
                        revert(0x1c, 0x04)
                    }
                }
                // Delete the approved address if any.
                if approvedAddress { sstore(add(1, ownershipSlot), 0) }
            }
            // Update with the new owner.
            sstore(ownershipSlot, xor(ownershipPacked, xor(from, to)))
            // Decrement the balance of `from`.
            {
                let fromBalanceSlot := keccak256(0x0c, 0x1c)
                sstore(fromBalanceSlot, sub(sload(fromBalanceSlot), 1))
            }
            // Increment the balance of `to`.
            {
                mstore(0x00, to)
                let toBalanceSlot := keccak256(0x0c, 0x1c)
                let toBalanceSlotPacked := add(sload(toBalanceSlot), 1)
                // Revert if `to` is the zero address, or if the account balance overflows.
                if iszero(mul(to, and(toBalanceSlotPacked, _MAX_ACCOUNT_BALANCE))) {
                    // `TransferToZeroAddress()`, `AccountBalanceOverflow()`.
                    mstore(shl(2, iszero(to)), 0xea553b3401336cea)
                    revert(0x1c, 0x04)
                }
                sstore(toBalanceSlot, toBalanceSlotPacked)
            }
            // Emit the {Transfer} event.
            log4(codesize(), 0x00, _TRANSFER_EVENT_SIGNATURE, from, to, id)
        }
        _afterTokenTransfer(from, to, id);
    }

    /// @dev Equivalent to `safeTransferFrom(from, to, id, "")`.
    function safeTransferFrom(address from, address to, uint256 id) public payable virtual {
        transferFrom(from, to, id);
        if (_hasCode(to)) _checkOnERC721Received(from, to, id, "");
    }

    /// @dev Transfers token `id` from `from` to `to`.
    ///
    /// Requirements:
    ///
    /// - Token `id` must exist.
    /// - `from` must be the owner of the token.
    /// - `to` cannot be the zero address.
    /// - The caller must be the owner of the token, or be approved to manage the token.
    /// - If `to` refers to a smart contract, it must implement
    ///   {IERC721Receiver-onERC721Received}, which is called upon a safe transfer.
    ///
    /// Emits a {Transfer} event.
    function safeTransferFrom(address from, address to, uint256 id, bytes calldata data)
        public
        payable
        virtual
    {
        transferFrom(from, to, id);
        if (_hasCode(to)) _checkOnERC721Received(from, to, id, data);
    }

    /// @dev Returns true if this contract implements the interface defined by `interfaceId`.
    /// See: https://eips.ethereum.org/EIPS/eip-165
    /// This function call must use less than 30000 gas.
    function supportsInterface(bytes4 interfaceId) public view virtual returns (bool result) {
        /// @solidity memory-safe-assembly
        assembly {
            let s := shr(224, interfaceId)
            // ERC165: 0x01ffc9a7, ERC721: 0x80ac58cd, ERC721Metadata: 0x5b5e139f.
            result := or(or(eq(s, 0x01ffc9a7), eq(s, 0x80ac58cd)), eq(s, 0x5b5e139f))
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                  INTERNAL QUERY FUNCTIONS                  */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Returns if token `id` exists.
    function _exists(uint256 id) internal view virtual returns (bool result) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, id)
            mstore(0x1c, _ERC721_MASTER_SLOT_SEED)
            result := iszero(iszero(shl(96, sload(add(id, add(id, keccak256(0x00, 0x20)))))))
        }
    }

    /// @dev Returns the owner of token `id`.
    /// Returns the zero address instead of reverting if the token does not exist.
    function _ownerOf(uint256 id) internal view virtual returns (address result) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, id)
            mstore(0x1c, _ERC721_MASTER_SLOT_SEED)
            result := shr(96, shl(96, sload(add(id, add(id, keccak256(0x00, 0x20))))))
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*            INTERNAL DATA HITCHHIKING FUNCTIONS             */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    // For performance, no events are emitted for the hitchhiking setters.
    // Please emit your own events if required.

    /// @dev Returns the auxiliary data for `owner`.
    /// Minting, transferring, burning the tokens of `owner` will not change the auxiliary data.
    /// Auxiliary data can be set for any address, even if it does not have any tokens.
    function _getAux(address owner) internal view virtual returns (uint224 result) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x1c, _ERC721_MASTER_SLOT_SEED)
            mstore(0x00, owner)
            result := shr(32, sload(keccak256(0x0c, 0x1c)))
        }
    }

    /// @dev Set the auxiliary data for `owner` to `value`.
    /// Minting, transferring, burning the tokens of `owner` will not change the auxiliary data.
    /// Auxiliary data can be set for any address, even if it does not have any tokens.
    function _setAux(address owner, uint224 value) internal virtual {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x1c, _ERC721_MASTER_SLOT_SEED)
            mstore(0x00, owner)
            let balanceSlot := keccak256(0x0c, 0x1c)
            let packed := sload(balanceSlot)
            sstore(balanceSlot, xor(packed, shl(32, xor(value, shr(32, packed)))))
        }
    }

    /// @dev Returns the extra data for token `id`.
    /// Minting, transferring, burning a token will not change the extra data.
    /// The extra data can be set on a non-existent token.
    function _getExtraData(uint256 id) internal view virtual returns (uint96 result) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, id)
            mstore(0x1c, _ERC721_MASTER_SLOT_SEED)
            result := shr(160, sload(add(id, add(id, keccak256(0x00, 0x20)))))
        }
    }

    /// @dev Sets the extra data for token `id` to `value`.
    /// Minting, transferring, burning a token will not change the extra data.
    /// The extra data can be set on a non-existent token.
    function _setExtraData(uint256 id, uint96 value) internal virtual {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, id)
            mstore(0x1c, _ERC721_MASTER_SLOT_SEED)
            let ownershipSlot := add(id, add(id, keccak256(0x00, 0x20)))
            let packed := sload(ownershipSlot)
            sstore(ownershipSlot, xor(packed, shl(160, xor(value, shr(160, packed)))))
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                  INTERNAL MINT FUNCTIONS                   */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Mints token `id` to `to`.
    ///
    /// Requirements:
    ///
    /// - Token `id` must not exist.
    /// - `to` cannot be the zero address.
    ///
    /// Emits a {Transfer} event.
    function _mint(address to, uint256 id) internal virtual {
        _beforeTokenTransfer(address(0), to, id);
        /// @solidity memory-safe-assembly
        assembly {
            // Clear the upper 96 bits.
            to := shr(96, shl(96, to))
            // Load the ownership data.
            mstore(0x00, id)
            mstore(0x1c, _ERC721_MASTER_SLOT_SEED)
            let ownershipSlot := add(id, add(id, keccak256(0x00, 0x20)))
            let ownershipPacked := sload(ownershipSlot)
            // Revert if the token already exists.
            if shl(96, ownershipPacked) {
                mstore(0x00, 0xc991cbb1) // `TokenAlreadyExists()`.
                revert(0x1c, 0x04)
            }
            // Update with the owner.
            sstore(ownershipSlot, or(ownershipPacked, to))
            // Increment the balance of the owner.
            {
                mstore(0x00, to)
                let balanceSlot := keccak256(0x0c, 0x1c)
                let balanceSlotPacked := add(sload(balanceSlot), 1)
                // Revert if `to` is the zero address, or if the account balance overflows.
                if iszero(mul(to, and(balanceSlotPacked, _MAX_ACCOUNT_BALANCE))) {
                    // `TransferToZeroAddress()`, `AccountBalanceOverflow()`.
                    mstore(shl(2, iszero(to)), 0xea553b3401336cea)
                    revert(0x1c, 0x04)
                }
                sstore(balanceSlot, balanceSlotPacked)
            }
            // Emit the {Transfer} event.
            log4(codesize(), 0x00, _TRANSFER_EVENT_SIGNATURE, 0, to, id)
        }
        _afterTokenTransfer(address(0), to, id);
    }

    /// @dev Mints token `id` to `to`, and updates the extra data for token `id` to `value`.
    /// Does NOT check if token `id` already exists (assumes `id` is auto-incrementing).
    ///
    /// Requirements:
    ///
    /// - `to` cannot be the zero address.
    ///
    /// Emits a {Transfer} event.
    function _mintAndSetExtraDataUnchecked(address to, uint256 id, uint96 value) internal virtual {
        _beforeTokenTransfer(address(0), to, id);
        /// @solidity memory-safe-assembly
        assembly {
            // Clear the upper 96 bits.
            to := shr(96, shl(96, to))
            // Update with the owner and extra data.
            mstore(0x00, id)
            mstore(0x1c, _ERC721_MASTER_SLOT_SEED)
            sstore(add(id, add(id, keccak256(0x00, 0x20))), or(shl(160, value), to))
            // Increment the balance of the owner.
            {
                mstore(0x00, to)
                let balanceSlot := keccak256(0x0c, 0x1c)
                let balanceSlotPacked := add(sload(balanceSlot), 1)
                // Revert if `to` is the zero address, or if the account balance overflows.
                if iszero(mul(to, and(balanceSlotPacked, _MAX_ACCOUNT_BALANCE))) {
                    // `TransferToZeroAddress()`, `AccountBalanceOverflow()`.
                    mstore(shl(2, iszero(to)), 0xea553b3401336cea)
                    revert(0x1c, 0x04)
                }
                sstore(balanceSlot, balanceSlotPacked)
            }
            // Emit the {Transfer} event.
            log4(codesize(), 0x00, _TRANSFER_EVENT_SIGNATURE, 0, to, id)
        }
        _afterTokenTransfer(address(0), to, id);
    }

    /// @dev Equivalent to `_safeMint(to, id, "")`.
    function _safeMint(address to, uint256 id) internal virtual {
        _safeMint(to, id, "");
    }

    /// @dev Mints token `id` to `to`.
    ///
    /// Requirements:
    ///
    /// - Token `id` must not exist.
    /// - `to` cannot be the zero address.
    /// - If `to` refers to a smart contract, it must implement
    ///   {IERC721Receiver-onERC721Received}, which is called upon a safe transfer.
    ///
    /// Emits a {Transfer} event.
    function _safeMint(address to, uint256 id, bytes memory data) internal virtual {
        _mint(to, id);
        if (_hasCode(to)) _checkOnERC721Received(address(0), to, id, data);
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                  INTERNAL BURN FUNCTIONS                   */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Equivalent to `_burn(address(0), id)`.
    function _burn(uint256 id) internal virtual {
        _burn(address(0), id);
    }

    /// @dev Destroys token `id`, using `by`.
    ///
    /// Requirements:
    ///
    /// - Token `id` must exist.
    /// - If `by` is not the zero address,
    ///   it must be the owner of the token, or be approved to manage the token.
    ///
    /// Emits a {Transfer} event.
    function _burn(address by, uint256 id) internal virtual {
        address owner = ownerOf(id);
        _beforeTokenTransfer(owner, address(0), id);
        /// @solidity memory-safe-assembly
        assembly {
            // Clear the upper 96 bits.
            by := shr(96, shl(96, by))
            // Load the ownership data.
            mstore(0x00, id)
            mstore(0x1c, or(_ERC721_MASTER_SLOT_SEED, by))
            let ownershipSlot := add(id, add(id, keccak256(0x00, 0x20)))
            let ownershipPacked := sload(ownershipSlot)
            // Reload the owner in case it is changed in `_beforeTokenTransfer`.
            owner := shr(96, shl(96, ownershipPacked))
            // Revert if the token does not exist.
            if iszero(owner) {
                mstore(0x00, 0xceea21b6) // `TokenDoesNotExist()`.
                revert(0x1c, 0x04)
            }
            // Load and check the token approval.
            {
                mstore(0x00, owner)
                let approvedAddress := sload(add(1, ownershipSlot))
                // If `by` is not the zero address, do the authorization check.
                // Revert if the `by` is not the owner, nor approved.
                if iszero(or(iszero(by), or(eq(by, owner), eq(by, approvedAddress)))) {
                    if iszero(sload(keccak256(0x0c, 0x30))) {
                        mstore(0x00, 0x4b6e7f18) // `NotOwnerNorApproved()`.
                        revert(0x1c, 0x04)
                    }
                }
                // Delete the approved address if any.
                if approvedAddress { sstore(add(1, ownershipSlot), 0) }
            }
            // Clear the owner.
            sstore(ownershipSlot, xor(ownershipPacked, owner))
            // Decrement the balance of `owner`.
            {
                let balanceSlot := keccak256(0x0c, 0x1c)
                sstore(balanceSlot, sub(sload(balanceSlot), 1))
            }
            // Emit the {Transfer} event.
            log4(codesize(), 0x00, _TRANSFER_EVENT_SIGNATURE, owner, 0, id)
        }
        _afterTokenTransfer(owner, address(0), id);
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                INTERNAL APPROVAL FUNCTIONS                 */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Returns whether `account` is the owner of token `id`, or is approved to manage it.
    ///
    /// Requirements:
    /// - Token `id` must exist.
    function _isApprovedOrOwner(address account, uint256 id)
        internal
        view
        virtual
        returns (bool result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            result := 1
            // Clear the upper 96 bits.
            account := shr(96, shl(96, account))
            // Load the ownership data.
            mstore(0x00, id)
            mstore(0x1c, or(_ERC721_MASTER_SLOT_SEED, account))
            let ownershipSlot := add(id, add(id, keccak256(0x00, 0x20)))
            let owner := shr(96, shl(96, sload(ownershipSlot)))
            // Revert if the token does not exist.
            if iszero(owner) {
                mstore(0x00, 0xceea21b6) // `TokenDoesNotExist()`.
                revert(0x1c, 0x04)
            }
            // Check if `account` is the `owner`.
            if iszero(eq(account, owner)) {
                mstore(0x00, owner)
                // Check if `account` is approved to manage the token.
                if iszero(sload(keccak256(0x0c, 0x30))) {
                    result := eq(account, sload(add(1, ownershipSlot)))
                }
            }
        }
    }

    /// @dev Returns the account approved to manage token `id`.
    /// Returns the zero address instead of reverting if the token does not exist.
    function _getApproved(uint256 id) internal view virtual returns (address result) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, id)
            mstore(0x1c, _ERC721_MASTER_SLOT_SEED)
            result := sload(add(1, add(id, add(id, keccak256(0x00, 0x20)))))
        }
    }

    /// @dev Equivalent to `_approve(address(0), account, id)`.
    function _approve(address account, uint256 id) internal virtual {
        _approve(address(0), account, id);
    }

    /// @dev Sets `account` as the approved account to manage token `id`, using `by`.
    ///
    /// Requirements:
    /// - Token `id` must exist.
    /// - If `by` is not the zero address, `by` must be the owner
    ///   or an approved operator for the token owner.
    ///
    /// Emits a {Approval} event.
    function _approve(address by, address account, uint256 id) internal virtual {
        /// @solidity memory-safe-assembly
        assembly {
            // Clear the upper 96 bits.
            let bitmaskAddress := shr(96, not(0))
            account := and(bitmaskAddress, account)
            by := and(bitmaskAddress, by)
            // Load the owner of the token.
            mstore(0x00, id)
            mstore(0x1c, or(_ERC721_MASTER_SLOT_SEED, by))
            let ownershipSlot := add(id, add(id, keccak256(0x00, 0x20)))
            let owner := and(bitmaskAddress, sload(ownershipSlot))
            // Revert if the token does not exist.
            if iszero(owner) {
                mstore(0x00, 0xceea21b6) // `TokenDoesNotExist()`.
                revert(0x1c, 0x04)
            }
            // If `by` is not the zero address, do the authorization check.
            // Revert if `by` is not the owner, nor approved.
            if iszero(or(iszero(by), eq(by, owner))) {
                mstore(0x00, owner)
                if iszero(sload(keccak256(0x0c, 0x30))) {
                    mstore(0x00, 0x4b6e7f18) // `NotOwnerNorApproved()`.
                    revert(0x1c, 0x04)
                }
            }
            // Sets `account` as the approved account to manage `id`.
            sstore(add(1, ownershipSlot), account)
            // Emit the {Approval} event.
            log4(codesize(), 0x00, _APPROVAL_EVENT_SIGNATURE, owner, account, id)
        }
    }

    /// @dev Approve or remove the `operator` as an operator for `by`,
    /// without authorization checks.
    ///
    /// Emits an {ApprovalForAll} event.
    function _setApprovalForAll(address by, address operator, bool isApproved) internal virtual {
        /// @solidity memory-safe-assembly
        assembly {
            // Clear the upper 96 bits.
            by := shr(96, shl(96, by))
            operator := shr(96, shl(96, operator))
            // Convert to 0 or 1.
            isApproved := iszero(iszero(isApproved))
            // Update the `isApproved` for (`by`, `operator`).
            mstore(0x1c, or(_ERC721_MASTER_SLOT_SEED, operator))
            mstore(0x00, by)
            sstore(keccak256(0x0c, 0x30), isApproved)
            // Emit the {ApprovalForAll} event.
            mstore(0x00, isApproved)
            log3(0x00, 0x20, _APPROVAL_FOR_ALL_EVENT_SIGNATURE, by, operator)
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                INTERNAL TRANSFER FUNCTIONS                 */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Equivalent to `_transfer(address(0), from, to, id)`.
    function _transfer(address from, address to, uint256 id) internal virtual {
        _transfer(address(0), from, to, id);
    }

    /// @dev Transfers token `id` from `from` to `to`.
    ///
    /// Requirements:
    ///
    /// - Token `id` must exist.
    /// - `from` must be the owner of the token.
    /// - `to` cannot be the zero address.
    /// - If `by` is not the zero address,
    ///   it must be the owner of the token, or be approved to manage the token.
    ///
    /// Emits a {Transfer} event.
    function _transfer(address by, address from, address to, uint256 id) internal virtual {
        _beforeTokenTransfer(from, to, id);
        /// @solidity memory-safe-assembly
        assembly {
            // Clear the upper 96 bits.
            let bitmaskAddress := shr(96, not(0))
            from := and(bitmaskAddress, from)
            to := and(bitmaskAddress, to)
            by := and(bitmaskAddress, by)
            // Load the ownership data.
            mstore(0x00, id)
            mstore(0x1c, or(_ERC721_MASTER_SLOT_SEED, by))
            let ownershipSlot := add(id, add(id, keccak256(0x00, 0x20)))
            let ownershipPacked := sload(ownershipSlot)
            let owner := and(bitmaskAddress, ownershipPacked)
            // Revert if the token does not exist, or if `from` is not the owner.
            if iszero(mul(owner, eq(owner, from))) {
                // `TokenDoesNotExist()`, `TransferFromIncorrectOwner()`.
                mstore(shl(2, iszero(owner)), 0xceea21b6a1148100)
                revert(0x1c, 0x04)
            }
            // Load, check, and update the token approval.
            {
                mstore(0x00, from)
                let approvedAddress := sload(add(1, ownershipSlot))
                // If `by` is not the zero address, do the authorization check.
                // Revert if the `by` is not the owner, nor approved.
                if iszero(or(iszero(by), or(eq(by, from), eq(by, approvedAddress)))) {
                    if iszero(sload(keccak256(0x0c, 0x30))) {
                        mstore(0x00, 0x4b6e7f18) // `NotOwnerNorApproved()`.
                        revert(0x1c, 0x04)
                    }
                }
                // Delete the approved address if any.
                if approvedAddress { sstore(add(1, ownershipSlot), 0) }
            }
            // Update with the new owner.
            sstore(ownershipSlot, xor(ownershipPacked, xor(from, to)))
            // Decrement the balance of `from`.
            {
                let fromBalanceSlot := keccak256(0x0c, 0x1c)
                sstore(fromBalanceSlot, sub(sload(fromBalanceSlot), 1))
            }
            // Increment the balance of `to`.
            {
                mstore(0x00, to)
                let toBalanceSlot := keccak256(0x0c, 0x1c)
                let toBalanceSlotPacked := add(sload(toBalanceSlot), 1)
                // Revert if `to` is the zero address, or if the account balance overflows.
                if iszero(mul(to, and(toBalanceSlotPacked, _MAX_ACCOUNT_BALANCE))) {
                    // `TransferToZeroAddress()`, `AccountBalanceOverflow()`.
                    mstore(shl(2, iszero(to)), 0xea553b3401336cea)
                    revert(0x1c, 0x04)
                }
                sstore(toBalanceSlot, toBalanceSlotPacked)
            }
            // Emit the {Transfer} event.
            log4(codesize(), 0x00, _TRANSFER_EVENT_SIGNATURE, from, to, id)
        }
        _afterTokenTransfer(from, to, id);
    }

    /// @dev Equivalent to `_safeTransfer(from, to, id, "")`.
    function _safeTransfer(address from, address to, uint256 id) internal virtual {
        _safeTransfer(from, to, id, "");
    }

    /// @dev Transfers token `id` from `from` to `to`.
    ///
    /// Requirements:
    ///
    /// - Token `id` must exist.
    /// - `from` must be the owner of the token.
    /// - `to` cannot be the zero address.
    /// - The caller must be the owner of the token, or be approved to manage the token.
    /// - If `to` refers to a smart contract, it must implement
    ///   {IERC721Receiver-onERC721Received}, which is called upon a safe transfer.
    ///
    /// Emits a {Transfer} event.
    function _safeTransfer(address from, address to, uint256 id, bytes memory data)
        internal
        virtual
    {
        _transfer(address(0), from, to, id);
        if (_hasCode(to)) _checkOnERC721Received(from, to, id, data);
    }

    /// @dev Equivalent to `_safeTransfer(by, from, to, id, "")`.
    function _safeTransfer(address by, address from, address to, uint256 id) internal virtual {
        _safeTransfer(by, from, to, id, "");
    }

    /// @dev Transfers token `id` from `from` to `to`.
    ///
    /// Requirements:
    ///
    /// - Token `id` must exist.
    /// - `from` must be the owner of the token.
    /// - `to` cannot be the zero address.
    /// - If `by` is not the zero address,
    ///   it must be the owner of the token, or be approved to manage the token.
    /// - If `to` refers to a smart contract, it must implement
    ///   {IERC721Receiver-onERC721Received}, which is called upon a safe transfer.
    ///
    /// Emits a {Transfer} event.
    function _safeTransfer(address by, address from, address to, uint256 id, bytes memory data)
        internal
        virtual
    {
        _transfer(by, from, to, id);
        if (_hasCode(to)) _checkOnERC721Received(from, to, id, data);
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                    HOOKS FOR OVERRIDING                    */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Hook that is called before any token transfers, including minting and burning.
    function _beforeTokenTransfer(address from, address to, uint256 id) internal virtual {}

    /// @dev Hook that is called after any token transfers, including minting and burning.
    function _afterTokenTransfer(address from, address to, uint256 id) internal virtual {}

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                      PRIVATE HELPERS                       */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Returns if `a` has bytecode of non-zero length.
    function _hasCode(address a) private view returns (bool result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := extcodesize(a) // Can handle dirty upper bits.
        }
    }

    /// @dev Perform a call to invoke {IERC721Receiver-onERC721Received} on `to`.
    /// Reverts if the target does not support the function correctly.
    function _checkOnERC721Received(address from, address to, uint256 id, bytes memory data)
        private
    {
        /// @solidity memory-safe-assembly
        assembly {
            // Prepare the calldata.
            let m := mload(0x40)
            let onERC721ReceivedSelector := 0x150b7a02
            mstore(m, onERC721ReceivedSelector)
            mstore(add(m, 0x20), caller()) // The `operator`, which is always `msg.sender`.
            mstore(add(m, 0x40), shr(96, shl(96, from)))
            mstore(add(m, 0x60), id)
            mstore(add(m, 0x80), 0x80)
            let n := mload(data)
            mstore(add(m, 0xa0), n)
            if n { pop(staticcall(gas(), 4, add(data, 0x20), n, add(m, 0xc0), n)) }
            // Revert if the call reverts.
            if iszero(call(gas(), to, 0, add(m, 0x1c), add(n, 0xa4), m, 0x20)) {
                if returndatasize() {
                    // Bubble up the revert if the call reverts.
                    returndatacopy(m, 0x00, returndatasize())
                    revert(m, returndatasize())
                }
            }
            // Load the returndata and compare it.
            if iszero(eq(mload(m), shl(224, onERC721ReceivedSelector))) {
                mstore(0x00, 0xd1a57ed6) // `TransferToNonERC721ReceiverImplementer()`.
                revert(0x1c, 0x04)
            }
        }
    }
}


// ===== FILE: lib/solady/src/tokens/ERC2981.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

/// @notice Simple ERC2981 NFT Royalty Standard implementation.
/// @author Solady (https://github.com/vectorized/solady/blob/main/src/tokens/ERC2981.sol)
/// @author Modified from OpenZeppelin (https://github.com/OpenZeppelin/openzeppelin-contracts/blob/master/contracts/token/common/ERC2981.sol)
abstract contract ERC2981 {
    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                       CUSTOM ERRORS                        */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev The royalty fee numerator exceeds the fee denominator.
    error RoyaltyOverflow();

    /// @dev The royalty receiver cannot be the zero address.
    error RoyaltyReceiverIsZeroAddress();

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                          STORAGE                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev The default royalty info is given by:
    /// ```
    ///     let packed := sload(_ERC2981_MASTER_SLOT_SEED)
    ///     let receiver := shr(96, packed)
    ///     let royaltyFraction := xor(packed, shl(96, receiver))
    /// ```
    ///
    /// The per token royalty info is given by.
    /// ```
    ///     mstore(0x00, tokenId)
    ///     mstore(0x20, _ERC2981_MASTER_SLOT_SEED)
    ///     let packed := sload(keccak256(0x00, 0x40))
    ///     let receiver := shr(96, packed)
    ///     let royaltyFraction := xor(packed, shl(96, receiver))
    /// ```
    uint256 private constant _ERC2981_MASTER_SLOT_SEED = 0xaa4ec00224afccfdb7;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                          ERC2981                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Checks that `_feeDenominator` is non-zero.
    constructor() {
        require(_feeDenominator() != 0, "Fee denominator cannot be zero.");
    }

    /// @dev Returns the denominator for the royalty amount.
    /// Defaults to 10000, which represents fees in basis points.
    /// Override this function to return a custom amount if needed.
    function _feeDenominator() internal pure virtual returns (uint96) {
        return 10000;
    }

    /// @dev Returns true if this contract implements the interface defined by `interfaceId`.
    /// See: https://eips.ethereum.org/EIPS/eip-165
    /// This function call must use less than 30000 gas.
    function supportsInterface(bytes4 interfaceId) public view virtual returns (bool result) {
        /// @solidity memory-safe-assembly
        assembly {
            let s := shr(224, interfaceId)
            // ERC165: 0x01ffc9a7, ERC2981: 0x2a55205a.
            result := or(eq(s, 0x01ffc9a7), eq(s, 0x2a55205a))
        }
    }

    /// @dev Returns the `receiver` and `royaltyAmount` for `tokenId` sold at `salePrice`.
    function royaltyInfo(uint256 tokenId, uint256 salePrice)
        public
        view
        virtual
        returns (address receiver, uint256 royaltyAmount)
    {
        uint256 feeDenominator = _feeDenominator();
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, tokenId)
            mstore(0x20, _ERC2981_MASTER_SLOT_SEED)
            let packed := sload(keccak256(0x00, 0x40))
            receiver := shr(96, packed)
            if iszero(receiver) {
                packed := sload(mload(0x20))
                receiver := shr(96, packed)
            }
            let x := salePrice
            let y := xor(packed, shl(96, receiver)) // `feeNumerator`.
            // Overflow check, equivalent to `require(y == 0 || x <= type(uint256).max / y)`.
            // Out-of-gas revert. Should not be triggered in practice, but included for safety.
            returndatacopy(returndatasize(), returndatasize(), mul(y, gt(x, div(not(0), y))))
            royaltyAmount := div(mul(x, y), feeDenominator)
        }
    }

    /// @dev Sets the default royalty `receiver` and `feeNumerator`.
    ///
    /// Requirements:
    /// - `receiver` must not be the zero address.
    /// - `feeNumerator` must not be greater than the fee denominator.
    function _setDefaultRoyalty(address receiver, uint96 feeNumerator) internal virtual {
        uint256 feeDenominator = _feeDenominator();
        /// @solidity memory-safe-assembly
        assembly {
            feeNumerator := shr(160, shl(160, feeNumerator))
            if gt(feeNumerator, feeDenominator) {
                mstore(0x00, 0x350a88b3) // `RoyaltyOverflow()`.
                revert(0x1c, 0x04)
            }
            let packed := shl(96, receiver)
            if iszero(packed) {
                mstore(0x00, 0xb4457eaa) // `RoyaltyReceiverIsZeroAddress()`.
                revert(0x1c, 0x04)
            }
            sstore(_ERC2981_MASTER_SLOT_SEED, or(packed, feeNumerator))
        }
    }

    /// @dev Sets the default royalty `receiver` and `feeNumerator` to zero.
    function _deleteDefaultRoyalty() internal virtual {
        /// @solidity memory-safe-assembly
        assembly {
            sstore(_ERC2981_MASTER_SLOT_SEED, 0)
        }
    }

    /// @dev Sets the royalty `receiver` and `feeNumerator` for `tokenId`.
    ///
    /// Requirements:
    /// - `receiver` must not be the zero address.
    /// - `feeNumerator` must not be greater than the fee denominator.
    function _setTokenRoyalty(uint256 tokenId, address receiver, uint96 feeNumerator)
        internal
        virtual
    {
        uint256 feeDenominator = _feeDenominator();
        /// @solidity memory-safe-assembly
        assembly {
            feeNumerator := shr(160, shl(160, feeNumerator))
            if gt(feeNumerator, feeDenominator) {
                mstore(0x00, 0x350a88b3) // `RoyaltyOverflow()`.
                revert(0x1c, 0x04)
            }
            let packed := shl(96, receiver)
            if iszero(packed) {
                mstore(0x00, 0xb4457eaa) // `RoyaltyReceiverIsZeroAddress()`.
                revert(0x1c, 0x04)
            }
            mstore(0x00, tokenId)
            mstore(0x20, _ERC2981_MASTER_SLOT_SEED)
            sstore(keccak256(0x00, 0x40), or(packed, feeNumerator))
        }
    }

    /// @dev Sets the royalty `receiver` and `feeNumerator` for `tokenId` to zero.
    function _resetTokenRoyalty(uint256 tokenId) internal virtual {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, tokenId)
            mstore(0x20, _ERC2981_MASTER_SLOT_SEED)
            sstore(keccak256(0x00, 0x40), 0)
        }
    }
}


// ===== FILE: lib/solady/src/auth/Ownable.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

/// @notice Simple single owner authorization mixin.
/// @author Solady (https://github.com/vectorized/solady/blob/main/src/auth/Ownable.sol)
///
/// @dev Note:
/// This implementation does NOT auto-initialize the owner to `msg.sender`.
/// You MUST call the `_initializeOwner` in the constructor / initializer.
///
/// While the ownable portion follows
/// [EIP-173](https://eips.ethereum.org/EIPS/eip-173) for compatibility,
/// the nomenclature for the 2-step ownership handover may be unique to this codebase.
abstract contract Ownable {
    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                       CUSTOM ERRORS                        */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev The caller is not authorized to call the function.
    error Unauthorized();

    /// @dev The `newOwner` cannot be the zero address.
    error NewOwnerIsZeroAddress();

    /// @dev The `pendingOwner` does not have a valid handover request.
    error NoHandoverRequest();

    /// @dev Cannot double-initialize.
    error AlreadyInitialized();

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                           EVENTS                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev The ownership is transferred from `oldOwner` to `newOwner`.
    /// This event is intentionally kept the same as OpenZeppelin's Ownable to be
    /// compatible with indexers and [EIP-173](https://eips.ethereum.org/EIPS/eip-173),
    /// despite it not being as lightweight as a single argument event.
    event OwnershipTransferred(address indexed oldOwner, address indexed newOwner);

    /// @dev An ownership handover to `pendingOwner` has been requested.
    event OwnershipHandoverRequested(address indexed pendingOwner);

    /// @dev The ownership handover to `pendingOwner` has been canceled.
    event OwnershipHandoverCanceled(address indexed pendingOwner);

    /// @dev `keccak256(bytes("OwnershipTransferred(address,address)"))`.
    uint256 private constant _OWNERSHIP_TRANSFERRED_EVENT_SIGNATURE =
        0x8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e0;

    /// @dev `keccak256(bytes("OwnershipHandoverRequested(address)"))`.
    uint256 private constant _OWNERSHIP_HANDOVER_REQUESTED_EVENT_SIGNATURE =
        0xdbf36a107da19e49527a7176a1babf963b4b0ff8cde35ee35d6cd8f1f9ac7e1d;

    /// @dev `keccak256(bytes("OwnershipHandoverCanceled(address)"))`.
    uint256 private constant _OWNERSHIP_HANDOVER_CANCELED_EVENT_SIGNATURE =
        0xfa7b8eab7da67f412cc9575ed43464468f9bfbae89d1675917346ca6d8fe3c92;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                          STORAGE                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev The owner slot is given by:
    /// `bytes32(~uint256(uint32(bytes4(keccak256("_OWNER_SLOT_NOT")))))`.
    /// It is intentionally chosen to be a high value
    /// to avoid collision with lower slots.
    /// The choice of manual storage layout is to enable compatibility
    /// with both regular and upgradeable contracts.
    bytes32 internal constant _OWNER_SLOT =
        0xffffffffffffffffffffffffffffffffffffffffffffffffffffffff74873927;

    /// The ownership handover slot of `newOwner` is given by:
    /// ```
    ///     mstore(0x00, or(shl(96, user), _HANDOVER_SLOT_SEED))
    ///     let handoverSlot := keccak256(0x00, 0x20)
    /// ```
    /// It stores the expiry timestamp of the two-step ownership handover.
    uint256 private constant _HANDOVER_SLOT_SEED = 0x389a75e1;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                     INTERNAL FUNCTIONS                     */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Override to return true to make `_initializeOwner` prevent double-initialization.
    function _guardInitializeOwner() internal pure virtual returns (bool guard) {}

    /// @dev Initializes the owner directly without authorization guard.
    /// This function must be called upon initialization,
    /// regardless of whether the contract is upgradeable or not.
    /// This is to enable generalization to both regular and upgradeable contracts,
    /// and to save gas in case the initial owner is not the caller.
    /// For performance reasons, this function will not check if there
    /// is an existing owner.
    function _initializeOwner(address newOwner) internal virtual {
        if (_guardInitializeOwner()) {
            /// @solidity memory-safe-assembly
            assembly {
                let ownerSlot := _OWNER_SLOT
                if sload(ownerSlot) {
                    mstore(0x00, 0x0dc149f0) // `AlreadyInitialized()`.
                    revert(0x1c, 0x04)
                }
                // Clean the upper 96 bits.
                newOwner := shr(96, shl(96, newOwner))
                // Store the new value.
                sstore(ownerSlot, or(newOwner, shl(255, iszero(newOwner))))
                // Emit the {OwnershipTransferred} event.
                log3(0, 0, _OWNERSHIP_TRANSFERRED_EVENT_SIGNATURE, 0, newOwner)
            }
        } else {
            /// @solidity memory-safe-assembly
            assembly {
                // Clean the upper 96 bits.
                newOwner := shr(96, shl(96, newOwner))
                // Store the new value.
                sstore(_OWNER_SLOT, newOwner)
                // Emit the {OwnershipTransferred} event.
                log3(0, 0, _OWNERSHIP_TRANSFERRED_EVENT_SIGNATURE, 0, newOwner)
            }
        }
    }

    /// @dev Sets the owner directly without authorization guard.
    function _setOwner(address newOwner) internal virtual {
        if (_guardInitializeOwner()) {
            /// @solidity memory-safe-assembly
            assembly {
                let ownerSlot := _OWNER_SLOT
                // Clean the upper 96 bits.
                newOwner := shr(96, shl(96, newOwner))
                // Emit the {OwnershipTransferred} event.
                log3(0, 0, _OWNERSHIP_TRANSFERRED_EVENT_SIGNATURE, sload(ownerSlot), newOwner)
                // Store the new value.
                sstore(ownerSlot, or(newOwner, shl(255, iszero(newOwner))))
            }
        } else {
            /// @solidity memory-safe-assembly
            assembly {
                let ownerSlot := _OWNER_SLOT
                // Clean the upper 96 bits.
                newOwner := shr(96, shl(96, newOwner))
                // Emit the {OwnershipTransferred} event.
                log3(0, 0, _OWNERSHIP_TRANSFERRED_EVENT_SIGNATURE, sload(ownerSlot), newOwner)
                // Store the new value.
                sstore(ownerSlot, newOwner)
            }
        }
    }

    /// @dev Throws if the sender is not the owner.
    function _checkOwner() internal view virtual {
        /// @solidity memory-safe-assembly
        assembly {
            // If the caller is not the stored owner, revert.
            if iszero(eq(caller(), sload(_OWNER_SLOT))) {
                mstore(0x00, 0x82b42900) // `Unauthorized()`.
                revert(0x1c, 0x04)
            }
        }
    }

    /// @dev Returns how long a two-step ownership handover is valid for in seconds.
    /// Override to return a different value if needed.
    /// Made internal to conserve bytecode. Wrap it in a public function if needed.
    function _ownershipHandoverValidFor() internal view virtual returns (uint64) {
        return 48 * 3600;
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                  PUBLIC UPDATE FUNCTIONS                   */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Allows the owner to transfer the ownership to `newOwner`.
    function transferOwnership(address newOwner) public payable virtual onlyOwner {
        /// @solidity memory-safe-assembly
        assembly {
            if iszero(shl(96, newOwner)) {
                mstore(0x00, 0x7448fbae) // `NewOwnerIsZeroAddress()`.
                revert(0x1c, 0x04)
            }
        }
        _setOwner(newOwner);
    }

    /// @dev Allows the owner to renounce their ownership.
    function renounceOwnership() public payable virtual onlyOwner {
        _setOwner(address(0));
    }

    /// @dev Request a two-step ownership handover to the caller.
    /// The request will automatically expire in 48 hours (172800 seconds) by default.
    function requestOwnershipHandover() public payable virtual {
        unchecked {
            uint256 expires = block.timestamp + _ownershipHandoverValidFor();
            /// @solidity memory-safe-assembly
            assembly {
                // Compute and set the handover slot to `expires`.
                mstore(0x0c, _HANDOVER_SLOT_SEED)
                mstore(0x00, caller())
                sstore(keccak256(0x0c, 0x20), expires)
                // Emit the {OwnershipHandoverRequested} event.
                log2(0, 0, _OWNERSHIP_HANDOVER_REQUESTED_EVENT_SIGNATURE, caller())
            }
        }
    }

    /// @dev Cancels the two-step ownership handover to the caller, if any.
    function cancelOwnershipHandover() public payable virtual {
        /// @solidity memory-safe-assembly
        assembly {
            // Compute and set the handover slot to 0.
            mstore(0x0c, _HANDOVER_SLOT_SEED)
            mstore(0x00, caller())
            sstore(keccak256(0x0c, 0x20), 0)
            // Emit the {OwnershipHandoverCanceled} event.
            log2(0, 0, _OWNERSHIP_HANDOVER_CANCELED_EVENT_SIGNATURE, caller())
        }
    }

    /// @dev Allows the owner to complete the two-step ownership handover to `pendingOwner`.
    /// Reverts if there is no existing ownership handover requested by `pendingOwner`.
    function completeOwnershipHandover(address pendingOwner) public payable virtual onlyOwner {
        /// @solidity memory-safe-assembly
        assembly {
            // Compute and set the handover slot to 0.
            mstore(0x0c, _HANDOVER_SLOT_SEED)
            mstore(0x00, pendingOwner)
            let handoverSlot := keccak256(0x0c, 0x20)
            // If the handover does not exist, or has expired.
            if gt(timestamp(), sload(handoverSlot)) {
                mstore(0x00, 0x6f5e8818) // `NoHandoverRequest()`.
                revert(0x1c, 0x04)
            }
            // Set the handover slot to 0.
            sstore(handoverSlot, 0)
        }
        _setOwner(pendingOwner);
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                   PUBLIC READ FUNCTIONS                    */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Returns the owner of the contract.
    function owner() public view virtual returns (address result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := sload(_OWNER_SLOT)
        }
    }

    /// @dev Returns the expiry timestamp for the two-step ownership handover to `pendingOwner`.
    function ownershipHandoverExpiresAt(address pendingOwner)
        public
        view
        virtual
        returns (uint256 result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            // Compute the handover slot.
            mstore(0x0c, _HANDOVER_SLOT_SEED)
            mstore(0x00, pendingOwner)
            // Load the handover slot.
            result := sload(keccak256(0x0c, 0x20))
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                         MODIFIERS                          */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Marks a function as only callable by the owner.
    modifier onlyOwner() virtual {
        _checkOwner();
        _;
    }
}


// ===== FILE: lib/solady/src/utils/LibString.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

import {LibBytes} from "./LibBytes.sol";

/// @notice Library for converting numbers into strings and other string operations.
/// @author Solady (https://github.com/vectorized/solady/blob/main/src/utils/LibString.sol)
/// @author Modified from Solmate (https://github.com/transmissions11/solmate/blob/main/src/utils/LibString.sol)
///
/// @dev Note:
/// For performance and bytecode compactness, most of the string operations are restricted to
/// byte strings (7-bit ASCII), except where otherwise specified.
/// Usage of byte string operations on charsets with runes spanning two or more bytes
/// can lead to undefined behavior.
library LibString {
    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                          STRUCTS                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Goated string storage struct that totally MOGs, no cap, fr.
    /// Uses less gas and bytecode than Solidity's native string storage. It's meta af.
    /// Packs length with the first 31 bytes if <255 bytes, so it’s mad tight.
    struct StringStorage {
        bytes32 _spacer;
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                        CUSTOM ERRORS                       */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev The length of the output is too small to contain all the hex digits.
    error HexLengthInsufficient();

    /// @dev The length of the string is more than 32 bytes.
    error TooBigForSmallString();

    /// @dev The input string must be a 7-bit ASCII.
    error StringNot7BitASCII();

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                         CONSTANTS                          */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev The constant returned when the `search` is not found in the string.
    uint256 internal constant NOT_FOUND = type(uint256).max;

    /// @dev Lookup for '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'.
    uint128 internal constant ALPHANUMERIC_7_BIT_ASCII = 0x7fffffe07fffffe03ff000000000000;

    /// @dev Lookup for 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'.
    uint128 internal constant LETTERS_7_BIT_ASCII = 0x7fffffe07fffffe0000000000000000;

    /// @dev Lookup for 'abcdefghijklmnopqrstuvwxyz'.
    uint128 internal constant LOWERCASE_7_BIT_ASCII = 0x7fffffe000000000000000000000000;

    /// @dev Lookup for 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.
    uint128 internal constant UPPERCASE_7_BIT_ASCII = 0x7fffffe0000000000000000;

    /// @dev Lookup for '0123456789'.
    uint128 internal constant DIGITS_7_BIT_ASCII = 0x3ff000000000000;

    /// @dev Lookup for '0123456789abcdefABCDEF'.
    uint128 internal constant HEXDIGITS_7_BIT_ASCII = 0x7e0000007e03ff000000000000;

    /// @dev Lookup for '01234567'.
    uint128 internal constant OCTDIGITS_7_BIT_ASCII = 0xff000000000000;

    /// @dev Lookup for '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'.
    uint128 internal constant PRINTABLE_7_BIT_ASCII = 0x7fffffffffffffffffffffff00003e00;

    /// @dev Lookup for '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'.
    uint128 internal constant PUNCTUATION_7_BIT_ASCII = 0x78000001f8000001fc00fffe00000000;

    /// @dev Lookup for ' \t\n\r\x0b\x0c'.
    uint128 internal constant WHITESPACE_7_BIT_ASCII = 0x100003e00;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                 STRING STORAGE OPERATIONS                  */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Sets the value of the string storage `$` to `s`.
    function set(StringStorage storage $, string memory s) internal {
        LibBytes.set(bytesStorage($), bytes(s));
    }

    /// @dev Sets the value of the string storage `$` to `s`.
    function setCalldata(StringStorage storage $, string calldata s) internal {
        LibBytes.setCalldata(bytesStorage($), bytes(s));
    }

    /// @dev Sets the value of the string storage `$` to the empty string.
    function clear(StringStorage storage $) internal {
        delete $._spacer;
    }

    /// @dev Returns whether the value stored is `$` is the empty string "".
    function isEmpty(StringStorage storage $) internal view returns (bool) {
        return uint256($._spacer) & 0xff == uint256(0);
    }

    /// @dev Returns the length of the value stored in `$`.
    function length(StringStorage storage $) internal view returns (uint256) {
        return LibBytes.length(bytesStorage($));
    }

    /// @dev Returns the value stored in `$`.
    function get(StringStorage storage $) internal view returns (string memory) {
        return string(LibBytes.get(bytesStorage($)));
    }

    /// @dev Returns the uint8 at index `i`. If out-of-bounds, returns 0.
    function uint8At(StringStorage storage $, uint256 i) internal view returns (uint8) {
        return LibBytes.uint8At(bytesStorage($), i);
    }

    /// @dev Helper to cast `$` to a `BytesStorage`.
    function bytesStorage(StringStorage storage $)
        internal
        pure
        returns (LibBytes.BytesStorage storage casted)
    {
        /// @solidity memory-safe-assembly
        assembly {
            casted.slot := $.slot
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                     DECIMAL OPERATIONS                     */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Returns the base 10 decimal representation of `value`.
    function toString(uint256 value) internal pure returns (string memory result) {
        /// @solidity memory-safe-assembly
        assembly {
            // The maximum value of a uint256 contains 78 digits (1 byte per digit), but
            // we allocate 0xa0 bytes to keep the free memory pointer 32-byte word aligned.
            // We will need 1 word for the trailing zeros padding, 1 word for the length,
            // and 3 words for a maximum of 78 digits.
            result := add(mload(0x40), 0x80)
            mstore(0x40, add(result, 0x20)) // Allocate memory.
            mstore(result, 0) // Zeroize the slot after the string.

            let end := result // Cache the end of the memory to calculate the length later.
            let w := not(0) // Tsk.
            // We write the string from rightmost digit to leftmost digit.
            // The following is essentially a do-while loop that also handles the zero case.
            for { let temp := value } 1 {} {
                result := add(result, w) // `sub(result, 1)`.
                // Store the character to the pointer.
                // The ASCII index of the '0' character is 48.
                mstore8(result, add(48, mod(temp, 10)))
                temp := div(temp, 10) // Keep dividing `temp` until zero.
                if iszero(temp) { break }
            }
            let n := sub(end, result)
            result := sub(result, 0x20) // Move the pointer 32 bytes back to make room for the length.
            mstore(result, n) // Store the length.
        }
    }

    /// @dev Returns the base 10 decimal representation of `value`.
    function toString(int256 value) internal pure returns (string memory result) {
        if (value >= 0) return toString(uint256(value));
        unchecked {
            result = toString(~uint256(value) + 1);
        }
        /// @solidity memory-safe-assembly
        assembly {
            // We still have some spare memory space on the left,
            // as we have allocated 3 words (96 bytes) for up to 78 digits.
            let n := mload(result) // Load the string length.
            mstore(result, 0x2d) // Store the '-' character.
            result := sub(result, 1) // Move back the string pointer by a byte.
            mstore(result, add(n, 1)) // Update the string length.
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                   HEXADECIMAL OPERATIONS                   */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Returns the hexadecimal representation of `value`,
    /// left-padded to an input length of `byteCount` bytes.
    /// The output is prefixed with "0x" encoded using 2 hexadecimal digits per byte,
    /// giving a total length of `byteCount * 2 + 2` bytes.
    /// Reverts if `byteCount` is too small for the output to contain all the digits.
    function toHexString(uint256 value, uint256 byteCount)
        internal
        pure
        returns (string memory result)
    {
        result = toHexStringNoPrefix(value, byteCount);
        /// @solidity memory-safe-assembly
        assembly {
            let n := add(mload(result), 2) // Compute the length.
            mstore(result, 0x3078) // Store the "0x" prefix.
            result := sub(result, 2) // Move the pointer.
            mstore(result, n) // Store the length.
        }
    }

    /// @dev Returns the hexadecimal representation of `value`,
    /// left-padded to an input length of `byteCount` bytes.
    /// The output is not prefixed with "0x" and is encoded using 2 hexadecimal digits per byte,
    /// giving a total length of `byteCount * 2` bytes.
    /// Reverts if `byteCount` is too small for the output to contain all the digits.
    function toHexStringNoPrefix(uint256 value, uint256 byteCount)
        internal
        pure
        returns (string memory result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            // We need 0x20 bytes for the trailing zeros padding, `byteCount * 2` bytes
            // for the digits, 0x02 bytes for the prefix, and 0x20 bytes for the length.
            // We add 0x20 to the total and round down to a multiple of 0x20.
            // (0x20 + 0x20 + 0x02 + 0x20) = 0x62.
            result := add(mload(0x40), and(add(shl(1, byteCount), 0x42), not(0x1f)))
            mstore(0x40, add(result, 0x20)) // Allocate memory.
            mstore(result, 0) // Zeroize the slot after the string.

            let end := result // Cache the end to calculate the length later.
            // Store "0123456789abcdef" in scratch space.
            mstore(0x0f, 0x30313233343536373839616263646566)

            let start := sub(result, add(byteCount, byteCount))
            let w := not(1) // Tsk.
            let temp := value
            // We write the string from rightmost digit to leftmost digit.
            // The following is essentially a do-while loop that also handles the zero case.
            for {} 1 {} {
                result := add(result, w) // `sub(result, 2)`.
                mstore8(add(result, 1), mload(and(temp, 15)))
                mstore8(result, mload(and(shr(4, temp), 15)))
                temp := shr(8, temp)
                if iszero(xor(result, start)) { break }
            }
            if temp {
                mstore(0x00, 0x2194895a) // `HexLengthInsufficient()`.
                revert(0x1c, 0x04)
            }
            let n := sub(end, result)
            result := sub(result, 0x20)
            mstore(result, n) // Store the length.
        }
    }

    /// @dev Returns the hexadecimal representation of `value`.
    /// The output is prefixed with "0x" and encoded using 2 hexadecimal digits per byte.
    /// As address are 20 bytes long, the output will left-padded to have
    /// a length of `20 * 2 + 2` bytes.
    function toHexString(uint256 value) internal pure returns (string memory result) {
        result = toHexStringNoPrefix(value);
        /// @solidity memory-safe-assembly
        assembly {
            let n := add(mload(result), 2) // Compute the length.
            mstore(result, 0x3078) // Store the "0x" prefix.
            result := sub(result, 2) // Move the pointer.
            mstore(result, n) // Store the length.
        }
    }

    /// @dev Returns the hexadecimal representation of `value`.
    /// The output is prefixed with "0x".
    /// The output excludes leading "0" from the `toHexString` output.
    /// `0x00: "0x0", 0x01: "0x1", 0x12: "0x12", 0x123: "0x123"`.
    function toMinimalHexString(uint256 value) internal pure returns (string memory result) {
        result = toHexStringNoPrefix(value);
        /// @solidity memory-safe-assembly
        assembly {
            let o := eq(byte(0, mload(add(result, 0x20))), 0x30) // Whether leading zero is present.
            let n := add(mload(result), 2) // Compute the length.
            mstore(add(result, o), 0x3078) // Store the "0x" prefix, accounting for leading zero.
            result := sub(add(result, o), 2) // Move the pointer, accounting for leading zero.
            mstore(result, sub(n, o)) // Store the length, accounting for leading zero.
        }
    }

    /// @dev Returns the hexadecimal representation of `value`.
    /// The output excludes leading "0" from the `toHexStringNoPrefix` output.
    /// `0x00: "0", 0x01: "1", 0x12: "12", 0x123: "123"`.
    function toMinimalHexStringNoPrefix(uint256 value)
        internal
        pure
        returns (string memory result)
    {
        result = toHexStringNoPrefix(value);
        /// @solidity memory-safe-assembly
        assembly {
            let o := eq(byte(0, mload(add(result, 0x20))), 0x30) // Whether leading zero is present.
            let n := mload(result) // Get the length.
            result := add(result, o) // Move the pointer, accounting for leading zero.
            mstore(result, sub(n, o)) // Store the length, accounting for leading zero.
        }
    }

    /// @dev Returns the hexadecimal representation of `value`.
    /// The output is encoded using 2 hexadecimal digits per byte.
    /// As address are 20 bytes long, the output will left-padded to have
    /// a length of `20 * 2` bytes.
    function toHexStringNoPrefix(uint256 value) internal pure returns (string memory result) {
        /// @solidity memory-safe-assembly
        assembly {
            // We need 0x20 bytes for the trailing zeros padding, 0x20 bytes for the length,
            // 0x02 bytes for the prefix, and 0x40 bytes for the digits.
            // The next multiple of 0x20 above (0x20 + 0x20 + 0x02 + 0x40) is 0xa0.
            result := add(mload(0x40), 0x80)
            mstore(0x40, add(result, 0x20)) // Allocate memory.
            mstore(result, 0) // Zeroize the slot after the string.

            let end := result // Cache the end to calculate the length later.
            mstore(0x0f, 0x30313233343536373839616263646566) // Store the "0123456789abcdef" lookup.

            let w := not(1) // Tsk.
            // We write the string from rightmost digit to leftmost digit.
            // The following is essentially a do-while loop that also handles the zero case.
            for { let temp := value } 1 {} {
                result := add(result, w) // `sub(result, 2)`.
                mstore8(add(result, 1), mload(and(temp, 15)))
                mstore8(result, mload(and(shr(4, temp), 15)))
                temp := shr(8, temp)
                if iszero(temp) { break }
            }
            let n := sub(end, result)
            result := sub(result, 0x20)
            mstore(result, n) // Store the length.
        }
    }

    /// @dev Returns the hexadecimal representation of `value`.
    /// The output is prefixed with "0x", encoded using 2 hexadecimal digits per byte,
    /// and the alphabets are capitalized conditionally according to
    /// https://eips.ethereum.org/EIPS/eip-55
    function toHexStringChecksummed(address value) internal pure returns (string memory result) {
        result = toHexString(value);
        /// @solidity memory-safe-assembly
        assembly {
            let mask := shl(6, div(not(0), 255)) // `0b010000000100000000 ...`
            let o := add(result, 0x22)
            let hashed := and(keccak256(o, 40), mul(34, mask)) // `0b10001000 ... `
            let t := shl(240, 136) // `0b10001000 << 240`
            for { let i := 0 } 1 {} {
                mstore(add(i, i), mul(t, byte(i, hashed)))
                i := add(i, 1)
                if eq(i, 20) { break }
            }
            mstore(o, xor(mload(o), shr(1, and(mload(0x00), and(mload(o), mask)))))
            o := add(o, 0x20)
            mstore(o, xor(mload(o), shr(1, and(mload(0x20), and(mload(o), mask)))))
        }
    }

    /// @dev Returns the hexadecimal representation of `value`.
    /// The output is prefixed with "0x" and encoded using 2 hexadecimal digits per byte.
    function toHexString(address value) internal pure returns (string memory result) {
        result = toHexStringNoPrefix(value);
        /// @solidity memory-safe-assembly
        assembly {
            let n := add(mload(result), 2) // Compute the length.
            mstore(result, 0x3078) // Store the "0x" prefix.
            result := sub(result, 2) // Move the pointer.
            mstore(result, n) // Store the length.
        }
    }

    /// @dev Returns the hexadecimal representation of `value`.
    /// The output is encoded using 2 hexadecimal digits per byte.
    function toHexStringNoPrefix(address value) internal pure returns (string memory result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := mload(0x40)
            // Allocate memory.
            // We need 0x20 bytes for the trailing zeros padding, 0x20 bytes for the length,
            // 0x02 bytes for the prefix, and 0x28 bytes for the digits.
            // The next multiple of 0x20 above (0x20 + 0x20 + 0x02 + 0x28) is 0x80.
            mstore(0x40, add(result, 0x80))
            mstore(0x0f, 0x30313233343536373839616263646566) // Store the "0123456789abcdef" lookup.

            result := add(result, 2)
            mstore(result, 40) // Store the length.
            let o := add(result, 0x20)
            mstore(add(o, 40), 0) // Zeroize the slot after the string.
            value := shl(96, value)
            // We write the string from rightmost digit to leftmost digit.
            // The following is essentially a do-while loop that also handles the zero case.
            for { let i := 0 } 1 {} {
                let p := add(o, add(i, i))
                let temp := byte(i, value)
                mstore8(add(p, 1), mload(and(temp, 15)))
                mstore8(p, mload(shr(4, temp)))
                i := add(i, 1)
                if eq(i, 20) { break }
            }
        }
    }

    /// @dev Returns the hex encoded string from the raw bytes.
    /// The output is encoded using 2 hexadecimal digits per byte.
    function toHexString(bytes memory raw) internal pure returns (string memory result) {
        result = toHexStringNoPrefix(raw);
        /// @solidity memory-safe-assembly
        assembly {
            let n := add(mload(result), 2) // Compute the length.
            mstore(result, 0x3078) // Store the "0x" prefix.
            result := sub(result, 2) // Move the pointer.
            mstore(result, n) // Store the length.
        }
    }

    /// @dev Returns the hex encoded string from the raw bytes.
    /// The output is encoded using 2 hexadecimal digits per byte.
    function toHexStringNoPrefix(bytes memory raw) internal pure returns (string memory result) {
        /// @solidity memory-safe-assembly
        assembly {
            let n := mload(raw)
            result := add(mload(0x40), 2) // Skip 2 bytes for the optional prefix.
            mstore(result, add(n, n)) // Store the length of the output.

            mstore(0x0f, 0x30313233343536373839616263646566) // Store the "0123456789abcdef" lookup.
            let o := add(result, 0x20)
            let end := add(raw, n)
            for {} iszero(eq(raw, end)) {} {
                raw := add(raw, 1)
                mstore8(add(o, 1), mload(and(mload(raw), 15)))
                mstore8(o, mload(and(shr(4, mload(raw)), 15)))
                o := add(o, 2)
            }
            mstore(o, 0) // Zeroize the slot after the string.
            mstore(0x40, add(o, 0x20)) // Allocate memory.
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                   RUNE STRING OPERATIONS                   */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Returns the number of UTF characters in the string.
    function runeCount(string memory s) internal pure returns (uint256 result) {
        /// @solidity memory-safe-assembly
        assembly {
            if mload(s) {
                mstore(0x00, div(not(0), 255))
                mstore(0x20, 0x0202020202020202020202020202020202020202020202020303030304040506)
                let o := add(s, 0x20)
                let end := add(o, mload(s))
                for { result := 1 } 1 { result := add(result, 1) } {
                    o := add(o, byte(0, mload(shr(250, mload(o)))))
                    if iszero(lt(o, end)) { break }
                }
            }
        }
    }

    /// @dev Returns if this string is a 7-bit ASCII string.
    /// (i.e. all characters codes are in [0..127])
    function is7BitASCII(string memory s) internal pure returns (bool result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := 1
            let mask := shl(7, div(not(0), 255))
            let n := mload(s)
            if n {
                let o := add(s, 0x20)
                let end := add(o, n)
                let last := mload(end)
                mstore(end, 0)
                for {} 1 {} {
                    if and(mask, mload(o)) {
                        result := 0
                        break
                    }
                    o := add(o, 0x20)
                    if iszero(lt(o, end)) { break }
                }
                mstore(end, last)
            }
        }
    }

    /// @dev Returns if this string is a 7-bit ASCII string,
    /// AND all characters are in the `allowed` lookup.
    /// Note: If `s` is empty, returns true regardless of `allowed`.
    function is7BitASCII(string memory s, uint128 allowed) internal pure returns (bool result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := 1
            if mload(s) {
                let allowed_ := shr(128, shl(128, allowed))
                let o := add(s, 0x20)
                for { let end := add(o, mload(s)) } 1 {} {
                    result := and(result, shr(byte(0, mload(o)), allowed_))
                    o := add(o, 1)
                    if iszero(and(result, lt(o, end))) { break }
                }
            }
        }
    }

    /// @dev Converts the bytes in the 7-bit ASCII string `s` to
    /// an allowed lookup for use in `is7BitASCII(s, allowed)`.
    /// To save runtime gas, you can cache the result in an immutable variable.
    function to7BitASCIIAllowedLookup(string memory s) internal pure returns (uint128 result) {
        /// @solidity memory-safe-assembly
        assembly {
            if mload(s) {
                let o := add(s, 0x20)
                for { let end := add(o, mload(s)) } 1 {} {
                    result := or(result, shl(byte(0, mload(o)), 1))
                    o := add(o, 1)
                    if iszero(lt(o, end)) { break }
                }
                if shr(128, result) {
                    mstore(0x00, 0xc9807e0d) // `StringNot7BitASCII()`.
                    revert(0x1c, 0x04)
                }
            }
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                   BYTE STRING OPERATIONS                   */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    // For performance and bytecode compactness, byte string operations are restricted
    // to 7-bit ASCII strings. All offsets are byte offsets, not UTF character offsets.
    // Usage of byte string operations on charsets with runes spanning two or more bytes
    // can lead to undefined behavior.

    /// @dev Returns `subject` all occurrences of `needle` replaced with `replacement`.
    function replace(string memory subject, string memory needle, string memory replacement)
        internal
        pure
        returns (string memory)
    {
        return string(LibBytes.replace(bytes(subject), bytes(needle), bytes(replacement)));
    }

    /// @dev Returns the byte index of the first location of `needle` in `subject`,
    /// needleing from left to right, starting from `from`.
    /// Returns `NOT_FOUND` (i.e. `type(uint256).max`) if the `needle` is not found.
    function indexOf(string memory subject, string memory needle, uint256 from)
        internal
        pure
        returns (uint256)
    {
        return LibBytes.indexOf(bytes(subject), bytes(needle), from);
    }

    /// @dev Returns the byte index of the first location of `needle` in `subject`,
    /// needleing from left to right.
    /// Returns `NOT_FOUND` (i.e. `type(uint256).max`) if the `needle` is not found.
    function indexOf(string memory subject, string memory needle) internal pure returns (uint256) {
        return LibBytes.indexOf(bytes(subject), bytes(needle), 0);
    }

    /// @dev Returns the byte index of the first location of `needle` in `subject`,
    /// needleing from right to left, starting from `from`.
    /// Returns `NOT_FOUND` (i.e. `type(uint256).max`) if the `needle` is not found.
    function lastIndexOf(string memory subject, string memory needle, uint256 from)
        internal
        pure
        returns (uint256)
    {
        return LibBytes.lastIndexOf(bytes(subject), bytes(needle), from);
    }

    /// @dev Returns the byte index of the first location of `needle` in `subject`,
    /// needleing from right to left.
    /// Returns `NOT_FOUND` (i.e. `type(uint256).max`) if the `needle` is not found.
    function lastIndexOf(string memory subject, string memory needle)
        internal
        pure
        returns (uint256)
    {
        return LibBytes.lastIndexOf(bytes(subject), bytes(needle), type(uint256).max);
    }

    /// @dev Returns true if `needle` is found in `subject`, false otherwise.
    function contains(string memory subject, string memory needle) internal pure returns (bool) {
        return LibBytes.contains(bytes(subject), bytes(needle));
    }

    /// @dev Returns whether `subject` starts with `needle`.
    function startsWith(string memory subject, string memory needle) internal pure returns (bool) {
        return LibBytes.startsWith(bytes(subject), bytes(needle));
    }

    /// @dev Returns whether `subject` ends with `needle`.
    function endsWith(string memory subject, string memory needle) internal pure returns (bool) {
        return LibBytes.endsWith(bytes(subject), bytes(needle));
    }

    /// @dev Returns `subject` repeated `times`.
    function repeat(string memory subject, uint256 times) internal pure returns (string memory) {
        return string(LibBytes.repeat(bytes(subject), times));
    }

    /// @dev Returns a copy of `subject` sliced from `start` to `end` (exclusive).
    /// `start` and `end` are byte offsets.
    function slice(string memory subject, uint256 start, uint256 end)
        internal
        pure
        returns (string memory)
    {
        return string(LibBytes.slice(bytes(subject), start, end));
    }

    /// @dev Returns a copy of `subject` sliced from `start` to the end of the string.
    /// `start` is a byte offset.
    function slice(string memory subject, uint256 start) internal pure returns (string memory) {
        return string(LibBytes.slice(bytes(subject), start, type(uint256).max));
    }

    /// @dev Returns all the indices of `needle` in `subject`.
    /// The indices are byte offsets.
    function indicesOf(string memory subject, string memory needle)
        internal
        pure
        returns (uint256[] memory)
    {
        return LibBytes.indicesOf(bytes(subject), bytes(needle));
    }

    /// @dev Returns an arrays of strings based on the `delimiter` inside of the `subject` string.
    function split(string memory subject, string memory delimiter)
        internal
        pure
        returns (string[] memory result)
    {
        bytes[] memory a = LibBytes.split(bytes(subject), bytes(delimiter));
        /// @solidity memory-safe-assembly
        assembly {
            result := a
        }
    }

    /// @dev Returns a concatenated string of `a` and `b`.
    /// Cheaper than `string.concat()` and does not de-align the free memory pointer.
    function concat(string memory a, string memory b) internal pure returns (string memory) {
        return string(LibBytes.concat(bytes(a), bytes(b)));
    }

    /// @dev Returns a copy of the string in either lowercase or UPPERCASE.
    /// WARNING! This function is only compatible with 7-bit ASCII strings.
    function toCase(string memory subject, bool toUpper)
        internal
        pure
        returns (string memory result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let n := mload(subject)
            if n {
                result := mload(0x40)
                let o := add(result, 0x20)
                let d := sub(subject, result)
                let flags := shl(add(70, shl(5, toUpper)), 0x3ffffff)
                for { let end := add(o, n) } 1 {} {
                    let b := byte(0, mload(add(d, o)))
                    mstore8(o, xor(and(shr(b, flags), 0x20), b))
                    o := add(o, 1)
                    if eq(o, end) { break }
                }
                mstore(result, n) // Store the length.
                mstore(o, 0) // Zeroize the slot after the string.
                mstore(0x40, add(o, 0x20)) // Allocate memory.
            }
        }
    }

    /// @dev Returns a string from a small bytes32 string.
    /// `s` must be null-terminated, or behavior will be undefined.
    function fromSmallString(bytes32 s) internal pure returns (string memory result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := mload(0x40)
            let n := 0
            for {} byte(n, s) { n := add(n, 1) } {} // Scan for '\0'.
            mstore(result, n) // Store the length.
            let o := add(result, 0x20)
            mstore(o, s) // Store the bytes of the string.
            mstore(add(o, n), 0) // Zeroize the slot after the string.
            mstore(0x40, add(result, 0x40)) // Allocate memory.
        }
    }

    /// @dev Returns the small string, with all bytes after the first null byte zeroized.
    function normalizeSmallString(bytes32 s) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            for {} byte(result, s) { result := add(result, 1) } {} // Scan for '\0'.
            mstore(0x00, s)
            mstore(result, 0x00)
            result := mload(0x00)
        }
    }

    /// @dev Returns the string as a normalized null-terminated small string.
    function toSmallString(string memory s) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := mload(s)
            if iszero(lt(result, 33)) {
                mstore(0x00, 0xec92f9a3) // `TooBigForSmallString()`.
                revert(0x1c, 0x04)
            }
            result := shl(shl(3, sub(32, result)), mload(add(s, result)))
        }
    }

    /// @dev Returns a lowercased copy of the string.
    /// WARNING! This function is only compatible with 7-bit ASCII strings.
    function lower(string memory subject) internal pure returns (string memory result) {
        result = toCase(subject, false);
    }

    /// @dev Returns an UPPERCASED copy of the string.
    /// WARNING! This function is only compatible with 7-bit ASCII strings.
    function upper(string memory subject) internal pure returns (string memory result) {
        result = toCase(subject, true);
    }

    /// @dev Escapes the string to be used within HTML tags.
    function escapeHTML(string memory s) internal pure returns (string memory result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := mload(0x40)
            let end := add(s, mload(s))
            let o := add(result, 0x20)
            // Store the bytes of the packed offsets and strides into the scratch space.
            // `packed = (stride << 5) | offset`. Max offset is 20. Max stride is 6.
            mstore(0x1f, 0x900094)
            mstore(0x08, 0xc0000000a6ab)
            // Store "&quot;&amp;&#39;&lt;&gt;" into the scratch space.
            mstore(0x00, shl(64, 0x2671756f743b26616d703b262333393b266c743b2667743b))
            for {} iszero(eq(s, end)) {} {
                s := add(s, 1)
                let c := and(mload(s), 0xff)
                // Not in `["\"","'","&","<",">"]`.
                if iszero(and(shl(c, 1), 0x500000c400000000)) {
                    mstore8(o, c)
                    o := add(o, 1)
                    continue
                }
                let t := shr(248, mload(c))
                mstore(o, mload(and(t, 0x1f)))
                o := add(o, shr(5, t))
            }
            mstore(o, 0) // Zeroize the slot after the string.
            mstore(result, sub(o, add(result, 0x20))) // Store the length.
            mstore(0x40, add(o, 0x20)) // Allocate memory.
        }
    }

    /// @dev Escapes the string to be used within double-quotes in a JSON.
    /// If `addDoubleQuotes` is true, the result will be enclosed in double-quotes.
    function escapeJSON(string memory s, bool addDoubleQuotes)
        internal
        pure
        returns (string memory result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            result := mload(0x40)
            let o := add(result, 0x20)
            if addDoubleQuotes {
                mstore8(o, 34)
                o := add(1, o)
            }
            // Store "\\u0000" in scratch space.
            // Store "0123456789abcdef" in scratch space.
            // Also, store `{0x08:"b", 0x09:"t", 0x0a:"n", 0x0c:"f", 0x0d:"r"}`.
            // into the scratch space.
            mstore(0x15, 0x5c75303030303031323334353637383961626364656662746e006672)
            // Bitmask for detecting `["\"","\\"]`.
            let e := or(shl(0x22, 1), shl(0x5c, 1))
            for { let end := add(s, mload(s)) } iszero(eq(s, end)) {} {
                s := add(s, 1)
                let c := and(mload(s), 0xff)
                if iszero(lt(c, 0x20)) {
                    if iszero(and(shl(c, 1), e)) {
                        // Not in `["\"","\\"]`.
                        mstore8(o, c)
                        o := add(o, 1)
                        continue
                    }
                    mstore8(o, 0x5c) // "\\".
                    mstore8(add(o, 1), c)
                    o := add(o, 2)
                    continue
                }
                if iszero(and(shl(c, 1), 0x3700)) {
                    // Not in `["\b","\t","\n","\f","\d"]`.
                    mstore8(0x1d, mload(shr(4, c))) // Hex value.
                    mstore8(0x1e, mload(and(c, 15))) // Hex value.
                    mstore(o, mload(0x19)) // "\\u00XX".
                    o := add(o, 6)
                    continue
                }
                mstore8(o, 0x5c) // "\\".
                mstore8(add(o, 1), mload(add(c, 8)))
                o := add(o, 2)
            }
            if addDoubleQuotes {
                mstore8(o, 34)
                o := add(1, o)
            }
            mstore(o, 0) // Zeroize the slot after the string.
            mstore(result, sub(o, add(result, 0x20))) // Store the length.
            mstore(0x40, add(o, 0x20)) // Allocate memory.
        }
    }

    /// @dev Escapes the string to be used within double-quotes in a JSON.
    function escapeJSON(string memory s) internal pure returns (string memory result) {
        result = escapeJSON(s, false);
    }

    /// @dev Encodes `s` so that it can be safely used in a URI,
    /// just like `encodeURIComponent` in JavaScript.
    /// See: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/encodeURIComponent
    /// See: https://datatracker.ietf.org/doc/html/rfc2396
    /// See: https://datatracker.ietf.org/doc/html/rfc3986
    function encodeURIComponent(string memory s) internal pure returns (string memory result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := mload(0x40)
            // Store "0123456789ABCDEF" in scratch space.
            // Uppercased to be consistent with JavaScript's implementation.
            mstore(0x0f, 0x30313233343536373839414243444546)
            let o := add(result, 0x20)
            for { let end := add(s, mload(s)) } iszero(eq(s, end)) {} {
                s := add(s, 1)
                let c := and(mload(s), 0xff)
                // If not in `[0-9A-Z-a-z-_.!~*'()]`.
                if iszero(and(1, shr(c, 0x47fffffe87fffffe03ff678200000000))) {
                    mstore8(o, 0x25) // '%'.
                    mstore8(add(o, 1), mload(and(shr(4, c), 15)))
                    mstore8(add(o, 2), mload(and(c, 15)))
                    o := add(o, 3)
                    continue
                }
                mstore8(o, c)
                o := add(o, 1)
            }
            mstore(result, sub(o, add(result, 0x20))) // Store the length.
            mstore(o, 0) // Zeroize the slot after the string.
            mstore(0x40, add(o, 0x20)) // Allocate memory.
        }
    }

    /// @dev Returns whether `a` equals `b`.
    function eq(string memory a, string memory b) internal pure returns (bool result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := eq(keccak256(add(a, 0x20), mload(a)), keccak256(add(b, 0x20), mload(b)))
        }
    }

    /// @dev Returns whether `a` equals `b`, where `b` is a null-terminated small string.
    function eqs(string memory a, bytes32 b) internal pure returns (bool result) {
        /// @solidity memory-safe-assembly
        assembly {
            // These should be evaluated on compile time, as far as possible.
            let m := not(shl(7, div(not(iszero(b)), 255))) // `0x7f7f ...`.
            let x := not(or(m, or(b, add(m, and(b, m)))))
            let r := shl(7, iszero(iszero(shr(128, x))))
            r := or(r, shl(6, iszero(iszero(shr(64, shr(r, x))))))
            r := or(r, shl(5, lt(0xffffffff, shr(r, x))))
            r := or(r, shl(4, lt(0xffff, shr(r, x))))
            r := or(r, shl(3, lt(0xff, shr(r, x))))
            // forgefmt: disable-next-item
            result := gt(eq(mload(a), add(iszero(x), xor(31, shr(3, r)))),
                xor(shr(add(8, r), b), shr(add(8, r), mload(add(a, 0x20)))))
        }
    }

    /// @dev Returns 0 if `a == b`, -1 if `a < b`, +1 if `a > b`.
    /// If `a` == b[:a.length]`, and `a.length < b.length`, returns -1.
    function cmp(string memory a, string memory b) internal pure returns (int256) {
        return LibBytes.cmp(bytes(a), bytes(b));
    }

    /// @dev Packs a single string with its length into a single word.
    /// Returns `bytes32(0)` if the length is zero or greater than 31.
    function packOne(string memory a) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            // We don't need to zero right pad the string,
            // since this is our own custom non-standard packing scheme.
            result :=
                mul(
                    // Load the length and the bytes.
                    mload(add(a, 0x1f)),
                    // `length != 0 && length < 32`. Abuses underflow.
                    // Assumes that the length is valid and within the block gas limit.
                    lt(sub(mload(a), 1), 0x1f)
                )
        }
    }

    /// @dev Unpacks a string packed using {packOne}.
    /// Returns the empty string if `packed` is `bytes32(0)`.
    /// If `packed` is not an output of {packOne}, the output behavior is undefined.
    function unpackOne(bytes32 packed) internal pure returns (string memory result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := mload(0x40) // Grab the free memory pointer.
            mstore(0x40, add(result, 0x40)) // Allocate 2 words (1 for the length, 1 for the bytes).
            mstore(result, 0) // Zeroize the length slot.
            mstore(add(result, 0x1f), packed) // Store the length and bytes.
            mstore(add(add(result, 0x20), mload(result)), 0) // Right pad with zeroes.
        }
    }

    /// @dev Packs two strings with their lengths into a single word.
    /// Returns `bytes32(0)` if combined length is zero or greater than 30.
    function packTwo(string memory a, string memory b) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let aLen := mload(a)
            // We don't need to zero right pad the strings,
            // since this is our own custom non-standard packing scheme.
            result :=
                mul(
                    or( // Load the length and the bytes of `a` and `b`.
                    shl(shl(3, sub(0x1f, aLen)), mload(add(a, aLen))), mload(sub(add(b, 0x1e), aLen))),
                    // `totalLen != 0 && totalLen < 31`. Abuses underflow.
                    // Assumes that the lengths are valid and within the block gas limit.
                    lt(sub(add(aLen, mload(b)), 1), 0x1e)
                )
        }
    }

    /// @dev Unpacks strings packed using {packTwo}.
    /// Returns the empty strings if `packed` is `bytes32(0)`.
    /// If `packed` is not an output of {packTwo}, the output behavior is undefined.
    function unpackTwo(bytes32 packed)
        internal
        pure
        returns (string memory resultA, string memory resultB)
    {
        /// @solidity memory-safe-assembly
        assembly {
            resultA := mload(0x40) // Grab the free memory pointer.
            resultB := add(resultA, 0x40)
            // Allocate 2 words for each string (1 for the length, 1 for the byte). Total 4 words.
            mstore(0x40, add(resultB, 0x40))
            // Zeroize the length slots.
            mstore(resultA, 0)
            mstore(resultB, 0)
            // Store the lengths and bytes.
            mstore(add(resultA, 0x1f), packed)
            mstore(add(resultB, 0x1f), mload(add(add(resultA, 0x20), mload(resultA))))
            // Right pad with zeroes.
            mstore(add(add(resultA, 0x20), mload(resultA)), 0)
            mstore(add(add(resultB, 0x20), mload(resultB)), 0)
        }
    }

    /// @dev Directly returns `a` without copying.
    function directReturn(string memory a) internal pure {
        /// @solidity memory-safe-assembly
        assembly {
            // Assumes that the string does not start from the scratch space.
            let retStart := sub(a, 0x20)
            let retUnpaddedSize := add(mload(a), 0x40)
            // Right pad with zeroes. Just in case the string is produced
            // by a method that doesn't zero right pad.
            mstore(add(retStart, retUnpaddedSize), 0)
            mstore(retStart, 0x20) // Store the return offset.
            // End the transaction, returning the string.
            return(retStart, and(not(0x1f), add(0x1f, retUnpaddedSize)))
        }
    }
}


// ===== FILE: lib/solady/src/utils/Base64.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

/// @notice Library to encode strings in Base64.
/// @author Solady (https://github.com/vectorized/solady/blob/main/src/utils/Base64.sol)
/// @author Modified from Solmate (https://github.com/transmissions11/solmate/blob/main/src/utils/Base64.sol)
/// @author Modified from (https://github.com/Brechtpd/base64/blob/main/base64.sol) by Brecht Devos - <brecht@loopring.org>.
library Base64 {
    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                    ENCODING / DECODING                     */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Encodes `data` using the base64 encoding described in RFC 4648.
    /// See: https://datatracker.ietf.org/doc/html/rfc4648
    /// @param fileSafe  Whether to replace '+' with '-' and '/' with '_'.
    /// @param noPadding Whether to strip away the padding.
    function encode(bytes memory data, bool fileSafe, bool noPadding)
        internal
        pure
        returns (string memory result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let dataLength := mload(data)

            if dataLength {
                // Multiply by 4/3 rounded up.
                // The `shl(2, ...)` is equivalent to multiplying by 4.
                let encodedLength := shl(2, div(add(dataLength, 2), 3))

                // Set `result` to point to the start of the free memory.
                result := mload(0x40)

                // Store the table into the scratch space.
                // Offsetted by -1 byte so that the `mload` will load the character.
                // We will rewrite the free memory pointer at `0x40` later with
                // the allocated size.
                // The magic constant 0x0670 will turn "-_" into "+/".
                mstore(0x1f, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef")
                mstore(0x3f, xor("ghijklmnopqrstuvwxyz0123456789-_", mul(iszero(fileSafe), 0x0670)))

                // Skip the first slot, which stores the length.
                let ptr := add(result, 0x20)
                let end := add(ptr, encodedLength)

                let dataEnd := add(add(0x20, data), dataLength)
                let dataEndValue := mload(dataEnd) // Cache the value at the `dataEnd` slot.
                mstore(dataEnd, 0x00) // Zeroize the `dataEnd` slot to clear dirty bits.

                // Run over the input, 3 bytes at a time.
                for {} 1 {} {
                    data := add(data, 3) // Advance 3 bytes.
                    let input := mload(data)

                    // Write 4 bytes. Optimized for fewer stack operations.
                    mstore8(0, mload(and(shr(18, input), 0x3F)))
                    mstore8(1, mload(and(shr(12, input), 0x3F)))
                    mstore8(2, mload(and(shr(6, input), 0x3F)))
                    mstore8(3, mload(and(input, 0x3F)))
                    mstore(ptr, mload(0x00))

                    ptr := add(ptr, 4) // Advance 4 bytes.
                    if iszero(lt(ptr, end)) { break }
                }
                mstore(dataEnd, dataEndValue) // Restore the cached value at `dataEnd`.
                mstore(0x40, add(end, 0x20)) // Allocate the memory.
                // Equivalent to `o = [0, 2, 1][dataLength % 3]`.
                let o := div(2, mod(dataLength, 3))
                // Offset `ptr` and pad with '='. We can simply write over the end.
                mstore(sub(ptr, o), shl(240, 0x3d3d))
                // Set `o` to zero if there is padding.
                o := mul(iszero(iszero(noPadding)), o)
                mstore(sub(ptr, o), 0) // Zeroize the slot after the string.
                mstore(result, sub(encodedLength, o)) // Store the length.
            }
        }
    }

    /// @dev Encodes `data` using the base64 encoding described in RFC 4648.
    /// Equivalent to `encode(data, false, false)`.
    function encode(bytes memory data) internal pure returns (string memory result) {
        result = encode(data, false, false);
    }

    /// @dev Encodes `data` using the base64 encoding described in RFC 4648.
    /// Equivalent to `encode(data, fileSafe, false)`.
    function encode(bytes memory data, bool fileSafe)
        internal
        pure
        returns (string memory result)
    {
        result = encode(data, fileSafe, false);
    }

    /// @dev Decodes base64 encoded `data`.
    ///
    /// Supports:
    /// - RFC 4648 (both standard and file-safe mode).
    /// - RFC 3501 (63: ',').
    ///
    /// Does not support:
    /// - Line breaks.
    ///
    /// Note: For performance reasons,
    /// this function will NOT revert on invalid `data` inputs.
    /// Outputs for invalid inputs will simply be undefined behaviour.
    /// It is the user's responsibility to ensure that the `data`
    /// is a valid base64 encoded string.
    function decode(string memory data) internal pure returns (bytes memory result) {
        /// @solidity memory-safe-assembly
        assembly {
            let dataLength := mload(data)

            if dataLength {
                let decodedLength := mul(shr(2, dataLength), 3)

                for {} 1 {} {
                    // If padded.
                    if iszero(and(dataLength, 3)) {
                        let t := xor(mload(add(data, dataLength)), 0x3d3d)
                        // forgefmt: disable-next-item
                        decodedLength := sub(
                            decodedLength,
                            add(iszero(byte(30, t)), iszero(byte(31, t)))
                        )
                        break
                    }
                    // If non-padded.
                    decodedLength := add(decodedLength, sub(and(dataLength, 3), 1))
                    break
                }
                result := mload(0x40)

                // Write the length of the bytes.
                mstore(result, decodedLength)

                // Skip the first slot, which stores the length.
                let ptr := add(result, 0x20)
                let end := add(ptr, decodedLength)

                // Load the table into the scratch space.
                // Constants are optimized for smaller bytecode with zero gas overhead.
                // `m` also doubles as the mask of the upper 6 bits.
                let m := 0xfc000000fc00686c7074787c8084888c9094989ca0a4a8acb0b4b8bcc0c4c8cc
                mstore(0x5b, m)
                mstore(0x3b, 0x04080c1014181c2024282c3034383c4044484c5054585c6064)
                mstore(0x1a, 0xf8fcf800fcd0d4d8dce0e4e8ecf0f4)

                for {} 1 {} {
                    // Read 4 bytes.
                    data := add(data, 4)
                    let input := mload(data)

                    // Write 3 bytes.
                    // forgefmt: disable-next-item
                    mstore(ptr, or(
                        and(m, mload(byte(28, input))),
                        shr(6, or(
                            and(m, mload(byte(29, input))),
                            shr(6, or(
                                and(m, mload(byte(30, input))),
                                shr(6, mload(byte(31, input)))
                            ))
                        ))
                    ))
                    ptr := add(ptr, 3)
                    if iszero(lt(ptr, end)) { break }
                }
                mstore(0x40, add(end, 0x20)) // Allocate the memory.
                mstore(end, 0) // Zeroize the slot after the bytes.
                mstore(0x60, 0) // Restore the zero slot.
            }
        }
    }
}


// ===== FILE: src/RuneRenderer.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {LibString} from "solady/utils/LibString.sol";

/// @title RuneRenderer — fully on-chain procedural rune SVG
/// @notice Deterministic generation from keccak256(tokenId). 12 palettes × 6 halos × 6 frames × procedural glyph.
contract RuneRenderer {
    using LibString for uint256;

    struct Palette {
        string name;
        string bg1;
        string bg2;
        string glow;
        string core;
        string ink;
    }

    function palette(uint8 i) internal pure returns (Palette memory p) {
        uint8 k = i % 12;
        if (k == 0)  return Palette("gold",     "#3d2818", "#0e0805", "#f5d99a", "#ff9020", "#6b5238");
        if (k == 1)  return Palette("cobalt",   "#1a2838", "#050810", "#cce6ff", "#4488ff", "#3a5878");
        if (k == 2)  return Palette("amethyst", "#2a1530", "#08040d", "#e9c4ff", "#cc44ff", "#553b6a");
        if (k == 3)  return Palette("jade",     "#143025", "#040a08", "#a5e6c8", "#22cc88", "#2f5645");
        if (k == 4)  return Palette("blood",    "#3a1015", "#0c0204", "#ffaaaa", "#cc2244", "#6a2030");
        if (k == 5)  return Palette("ash",      "#1a1a1f", "#070708", "#dddde0", "#888892", "#444450");
        if (k == 6)  return Palette("rust",     "#3a1808", "#0d0502", "#ffb070", "#cc4411", "#6a3015");
        if (k == 7)  return Palette("frost",    "#1a2a30", "#040808", "#bce6f0", "#66bbcc", "#3a5060");
        if (k == 8)  return Palette("ember",    "#301008", "#0a0302", "#ffd088", "#ff5522", "#552010");
        if (k == 9)  return Palette("void",     "#0a0a14", "#020205", "#ccccff", "#5555aa", "#2a2a4a");
        if (k == 10) return Palette("moss",     "#1a2810", "#040805", "#c8e6a0", "#88aa33", "#445520");
        return            Palette("wine",     "#2a0820", "#08020a", "#ee99cc", "#aa3377", "#552244");
    }

    /// @notice main entry — returns full SVG for a token
    function svg(uint256 id) external pure returns (string memory) {
        bytes32 seed = keccak256(abi.encodePacked(id));
        Palette memory p = palette(uint8(seed[0]));
        string memory ids = id.toString();

        string memory head = string.concat(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">',
            _defs(ids, p, seed),
            '<rect width="400" height="400" fill="url(#bg', ids, ')"/>'
        );
        string memory bg = string.concat(
            '<rect width="400" height="400" fill="url(#bg', ids, ')" filter="url(#gr', ids, ')" opacity="0.5"/>'
        );
        string memory deco = string.concat(
            _frame(p, uint8(seed[2])),
            _halo(p, uint8(seed[1])),
            _innerCircles(p, uint8(seed[7]), uint8(seed[10]))
        );
        string memory body = string.concat(
            _glyph(p, seed),
            _label(id, p, uint8(seed[15])),
            '</svg>'
        );
        return string.concat(head, bg, deco, body);
    }

    function _defs(string memory ids, Palette memory p, bytes32 seed) internal pure returns (string memory) {
        uint256 mode14 = uint256(uint8(seed[14])) % 4;
        string memory freq;
        if (mode14 == 0) freq = "0.7";
        else if (mode14 == 1) freq = "0.8";
        else if (mode14 == 2) freq = "0.9";
        else freq = "1.0";
        string memory grad = string.concat(
            '<radialGradient id="bg', ids,
            '"><stop offset="0" stop-color="', p.bg1,
            '"/><stop offset="1" stop-color="', p.bg2,
            '"/></radialGradient>'
        );
        string memory filt = string.concat(
            '<filter id="gr', ids, '"><feTurbulence baseFrequency="', freq,
            '" numOctaves="2"/><feColorMatrix values="0 0 0 0 0.1  0 0 0 0 0.1  0 0 0 0 0.1  0 0 0 0.35 0"/></filter>'
        );
        return string.concat('<defs>', grad, filt, '</defs>');
    }

    function _halo(Palette memory p, uint8 b) internal pure returns (string memory) {
        uint256 k = uint256(b) % 6;
        if (k == 0) return ""; // none
        if (k == 1) return string.concat('<circle cx="200" cy="200" r="130" fill="none" stroke="', p.glow, '" stroke-width="0.8" opacity="0.3"/>');
        if (k == 2) return string.concat(
            '<circle cx="200" cy="200" r="130" fill="none" stroke="', p.glow, '" stroke-width="0.8" opacity="0.35"/>',
            '<circle cx="200" cy="200" r="105" fill="none" stroke="', p.glow, '" stroke-width="0.5" opacity="0.5"/>'
        );
        if (k == 3) return string.concat(
            '<circle cx="200" cy="200" r="140" fill="none" stroke="', p.glow, '" stroke-width="0.6" opacity="0.25"/>',
            '<circle cx="200" cy="200" r="115" fill="none" stroke="', p.glow, '" stroke-width="0.5" opacity="0.4"/>',
            '<circle cx="200" cy="200" r="90" fill="none" stroke="', p.glow, '" stroke-width="0.4" opacity="0.55"/>'
        );
        if (k == 4) return string.concat(
            '<g fill="none" stroke="', p.glow, '" stroke-width="0.6" opacity="0.4">',
            '<circle cx="200" cy="200" r="130"/>',
            '<circle cx="200" cy="70" r="3"/><circle cx="200" cy="330" r="3"/>',
            '<circle cx="70" cy="200" r="3"/><circle cx="330" cy="200" r="3"/></g>'
        );
        return string.concat( // broken
            '<g fill="none" stroke="', p.glow, '" stroke-width="0.7" opacity="0.4" stroke-dasharray="20 8">',
            '<circle cx="200" cy="200" r="130"/></g>'
        );
    }

    function _frame(Palette memory p, uint8 b) internal pure returns (string memory) {
        uint256 k = uint256(b) % 6;
        if (k == 0) return "";
        if (k == 1) return string.concat('<polygon points="200,55 350,310 50,310" fill="none" stroke="', p.core, '" stroke-width="0.5" opacity="0.3"/>');
        if (k == 2) return string.concat('<polygon points="200,60 330,135 330,265 200,340 70,265 70,135" fill="none" stroke="', p.core, '" stroke-width="0.5" opacity="0.3"/>');
        if (k == 3) return string.concat('<rect x="70" y="70" width="260" height="260" fill="none" stroke="', p.core, '" stroke-width="0.5" opacity="0.3"/>');
        if (k == 4) return string.concat('<polygon points="200,55 345,200 200,345 55,200" fill="none" stroke="', p.core, '" stroke-width="0.5" opacity="0.3"/>');
        return string.concat('<circle cx="200" cy="200" r="155" fill="none" stroke="', p.core, '" stroke-width="0.5" opacity="0.3"/>');
    }

    function _innerCircles(Palette memory p, uint8 nRaw, uint8 sizeByte) internal pure returns (string memory) {
        uint256 n = uint256(nRaw) % 3;
        if (n == 0) return "";
        if (n == 1) {
            uint256 r = 50 + (uint256(sizeByte) % 20);
            return string.concat('<circle cx="200" cy="200" r="', r.toString(), '" fill="none" stroke="', p.ink, '" stroke-width="0.4" opacity="0.5"/>');
        }
        return string.concat(
            '<g fill="none" stroke="', p.ink, '" stroke-width="0.4" opacity="0.4">',
            '<circle cx="200" cy="200" r="40"/><circle cx="200" cy="200" r="65"/></g>'
        );
    }

    function _glyph(Palette memory p, bytes32 seed) internal pure returns (string memory) {
        uint256 strokeCount = 3 + (uint256(uint8(seed[3])) % 4); // 3-6
        uint256 strokeW = 4 + (uint256(uint8(seed[6])) % 4); // 4-7
        string memory rot = _rotation(uint256(uint8(seed[5])) % 4);

        string memory strokes = _strokes(seed, strokeCount);
        string memory orb = _orb(p, uint256(uint8(seed[4])) % 20);

        return string.concat(
            '<g transform="rotate(', rot, ' 200 200)">',
            '<g stroke="', p.glow, '" stroke-width="', strokeW.toString(),
            '" fill="none" stroke-linecap="round" filter="drop-shadow(0 0 8px ', p.core, ')">',
            '<path d="', strokes, '"/></g>',
            orb,
            '</g>'
        );
    }

    function _rotation(uint256 mode) internal pure returns (string memory) {
        if (mode == 0) return "-22";
        if (mode == 1) return "-7";
        if (mode == 2) return "8";
        return "23";
    }

    function _strokes(bytes32 seed, uint256 count) internal pure returns (string memory out) {
        // 3 cols × 5 rows grid
        uint256[3] memory COLS = [uint256(160), 200, 240];
        uint256[5] memory ROWS = [uint256(130), 165, 200, 235, 270];
        for (uint256 i = 0; i < count; i++) {
            uint256 b1 = uint256(uint8(seed[8 + i * 2]));
            uint256 b2 = uint256(uint8(seed[9 + i * 2]));
            uint256 x1 = COLS[b1 % 3];
            uint256 y1 = ROWS[(b1 >> 2) % 5];
            uint256 x2 = COLS[b2 % 3];
            uint256 y2 = ROWS[(b2 >> 2) % 5];
            if (x1 == x2 && y1 == y2) continue;
            out = string.concat(
                out,
                "M", x1.toString(), " ", y1.toString(),
                " L", x2.toString(), " ", y2.toString(), " "
            );
        }
    }

    function _orb(Palette memory p, uint256 size) internal pure returns (string memory) {
        if (size <= 6) return "";
        uint256 r = size / 2;
        return string.concat('<circle cx="200" cy="200" r="', r.toString(), '" fill="', p.core, '" opacity="0.55"/>');
    }

    string private constant RUNES = unicode"ᚦᚱᚷᚹᚺᚾᛁᛃᛇᛈᛉᛊᛏᛒᛖᛗᛚᛜᛞᛟᚠᚢᚨᚲ";

    function _label(uint256 id, Palette memory p, uint8 runeIdx) internal pure returns (string memory) {
        // Each Elder Futhark rune is 3 bytes in UTF-8
        uint256 k = uint256(runeIdx) % 24;
        bytes memory all = bytes(RUNES);
        bytes memory ch = new bytes(3);
        ch[0] = all[k * 3];
        ch[1] = all[k * 3 + 1];
        ch[2] = all[k * 3 + 2];
        return string.concat(
            '<text x="200" y="375" text-anchor="middle" fill="', p.ink, '" font-family="monospace" font-size="9">RUNE / ',
            _pad4(id), ' / ', string(ch), ' / ', p.name,
            '</text>'
        );
    }

    function _pad4(uint256 n) internal pure returns (string memory) {
        string memory s = n.toString();
        if (n < 10) return string.concat("000", s);
        if (n < 100) return string.concat("00", s);
        if (n < 1000) return string.concat("0", s);
        return s;
    }
}


// ===== FILE: lib/solady/src/utils/LibBytes.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

/// @notice Library for byte related operations.
/// @author Solady (https://github.com/vectorized/solady/blob/main/src/utils/LibBytes.sol)
library LibBytes {
    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                          STRUCTS                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Goated bytes storage struct that totally MOGs, no cap, fr.
    /// Uses less gas and bytecode than Solidity's native bytes storage. It's meta af.
    /// Packs length with the first 31 bytes if <255 bytes, so it’s mad tight.
    struct BytesStorage {
        bytes32 _spacer;
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                         CONSTANTS                          */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev The constant returned when the `search` is not found in the bytes.
    uint256 internal constant NOT_FOUND = type(uint256).max;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                  BYTE STORAGE OPERATIONS                   */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Sets the value of the bytes storage `$` to `s`.
    function set(BytesStorage storage $, bytes memory s) internal {
        /// @solidity memory-safe-assembly
        assembly {
            let n := mload(s)
            let packed := or(0xff, shl(8, n))
            for { let i := 0 } 1 {} {
                if iszero(gt(n, 0xfe)) {
                    i := 0x1f
                    packed := or(n, shl(8, mload(add(s, i))))
                    if iszero(gt(n, i)) { break }
                }
                let o := add(s, 0x20)
                mstore(0x00, $.slot)
                for { let p := keccak256(0x00, 0x20) } 1 {} {
                    sstore(add(p, shr(5, i)), mload(add(o, i)))
                    i := add(i, 0x20)
                    if iszero(lt(i, n)) { break }
                }
                break
            }
            sstore($.slot, packed)
        }
    }

    /// @dev Sets the value of the bytes storage `$` to `s`.
    function setCalldata(BytesStorage storage $, bytes calldata s) internal {
        /// @solidity memory-safe-assembly
        assembly {
            let packed := or(0xff, shl(8, s.length))
            for { let i := 0 } 1 {} {
                if iszero(gt(s.length, 0xfe)) {
                    i := 0x1f
                    packed := or(s.length, shl(8, shr(8, calldataload(s.offset))))
                    if iszero(gt(s.length, i)) { break }
                }
                mstore(0x00, $.slot)
                for { let p := keccak256(0x00, 0x20) } 1 {} {
                    sstore(add(p, shr(5, i)), calldataload(add(s.offset, i)))
                    i := add(i, 0x20)
                    if iszero(lt(i, s.length)) { break }
                }
                break
            }
            sstore($.slot, packed)
        }
    }

    /// @dev Sets the value of the bytes storage `$` to the empty bytes.
    function clear(BytesStorage storage $) internal {
        delete $._spacer;
    }

    /// @dev Returns whether the value stored is `$` is the empty bytes "".
    function isEmpty(BytesStorage storage $) internal view returns (bool) {
        return uint256($._spacer) & 0xff == uint256(0);
    }

    /// @dev Returns the length of the value stored in `$`.
    function length(BytesStorage storage $) internal view returns (uint256 result) {
        result = uint256($._spacer);
        /// @solidity memory-safe-assembly
        assembly {
            let n := and(0xff, result)
            result := or(mul(shr(8, result), eq(0xff, n)), mul(n, iszero(eq(0xff, n))))
        }
    }

    /// @dev Returns the value stored in `$`.
    function get(BytesStorage storage $) internal view returns (bytes memory result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := mload(0x40)
            let o := add(result, 0x20)
            let packed := sload($.slot)
            let n := shr(8, packed)
            for { let i := 0 } 1 {} {
                if iszero(eq(or(packed, 0xff), packed)) {
                    mstore(o, packed)
                    n := and(0xff, packed)
                    i := 0x1f
                    if iszero(gt(n, i)) { break }
                }
                mstore(0x00, $.slot)
                for { let p := keccak256(0x00, 0x20) } 1 {} {
                    mstore(add(o, i), sload(add(p, shr(5, i))))
                    i := add(i, 0x20)
                    if iszero(lt(i, n)) { break }
                }
                break
            }
            mstore(result, n) // Store the length of the memory.
            mstore(add(o, n), 0) // Zeroize the slot after the bytes.
            mstore(0x40, add(add(o, n), 0x20)) // Allocate memory.
        }
    }

    /// @dev Returns the uint8 at index `i`. If out-of-bounds, returns 0.
    function uint8At(BytesStorage storage $, uint256 i) internal view returns (uint8 result) {
        /// @solidity memory-safe-assembly
        assembly {
            for { let packed := sload($.slot) } 1 {} {
                if iszero(eq(or(packed, 0xff), packed)) {
                    if iszero(gt(i, 0x1e)) {
                        result := byte(i, packed)
                        break
                    }
                    if iszero(gt(i, and(0xff, packed))) {
                        mstore(0x00, $.slot)
                        let j := sub(i, 0x1f)
                        result := byte(and(j, 0x1f), sload(add(keccak256(0x00, 0x20), shr(5, j))))
                    }
                    break
                }
                if iszero(gt(i, shr(8, packed))) {
                    mstore(0x00, $.slot)
                    result := byte(and(i, 0x1f), sload(add(keccak256(0x00, 0x20), shr(5, i))))
                }
                break
            }
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                      BYTES OPERATIONS                      */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Returns `subject` all occurrences of `needle` replaced with `replacement`.
    function replace(bytes memory subject, bytes memory needle, bytes memory replacement)
        internal
        pure
        returns (bytes memory result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            result := mload(0x40)
            let needleLen := mload(needle)
            let replacementLen := mload(replacement)
            let d := sub(result, subject) // Memory difference.
            let i := add(subject, 0x20) // Subject bytes pointer.
            mstore(0x00, add(i, mload(subject))) // End of subject.
            if iszero(gt(needleLen, mload(subject))) {
                let subjectSearchEnd := add(sub(mload(0x00), needleLen), 1)
                let h := 0 // The hash of `needle`.
                if iszero(lt(needleLen, 0x20)) { h := keccak256(add(needle, 0x20), needleLen) }
                let s := mload(add(needle, 0x20))
                for { let m := shl(3, sub(0x20, and(needleLen, 0x1f))) } 1 {} {
                    let t := mload(i)
                    // Whether the first `needleLen % 32` bytes of `subject` and `needle` matches.
                    if iszero(shr(m, xor(t, s))) {
                        if h {
                            if iszero(eq(keccak256(i, needleLen), h)) {
                                mstore(add(i, d), t)
                                i := add(i, 1)
                                if iszero(lt(i, subjectSearchEnd)) { break }
                                continue
                            }
                        }
                        // Copy the `replacement` one word at a time.
                        for { let j := 0 } 1 {} {
                            mstore(add(add(i, d), j), mload(add(add(replacement, 0x20), j)))
                            j := add(j, 0x20)
                            if iszero(lt(j, replacementLen)) { break }
                        }
                        d := sub(add(d, replacementLen), needleLen)
                        if needleLen {
                            i := add(i, needleLen)
                            if iszero(lt(i, subjectSearchEnd)) { break }
                            continue
                        }
                    }
                    mstore(add(i, d), t)
                    i := add(i, 1)
                    if iszero(lt(i, subjectSearchEnd)) { break }
                }
            }
            let end := mload(0x00)
            let n := add(sub(d, add(result, 0x20)), end)
            // Copy the rest of the bytes one word at a time.
            for {} lt(i, end) { i := add(i, 0x20) } { mstore(add(i, d), mload(i)) }
            let o := add(i, d)
            mstore(o, 0) // Zeroize the slot after the bytes.
            mstore(0x40, add(o, 0x20)) // Allocate memory.
            mstore(result, n) // Store the length.
        }
    }

    /// @dev Returns the byte index of the first location of `needle` in `subject`,
    /// needleing from left to right, starting from `from`.
    /// Returns `NOT_FOUND` (i.e. `type(uint256).max`) if the `needle` is not found.
    function indexOf(bytes memory subject, bytes memory needle, uint256 from)
        internal
        pure
        returns (uint256 result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            result := not(0) // Initialize to `NOT_FOUND`.
            for { let subjectLen := mload(subject) } 1 {} {
                if iszero(mload(needle)) {
                    result := from
                    if iszero(gt(from, subjectLen)) { break }
                    result := subjectLen
                    break
                }
                let needleLen := mload(needle)
                let subjectStart := add(subject, 0x20)

                subject := add(subjectStart, from)
                let end := add(sub(add(subjectStart, subjectLen), needleLen), 1)
                let m := shl(3, sub(0x20, and(needleLen, 0x1f)))
                let s := mload(add(needle, 0x20))

                if iszero(and(lt(subject, end), lt(from, subjectLen))) { break }

                if iszero(lt(needleLen, 0x20)) {
                    for { let h := keccak256(add(needle, 0x20), needleLen) } 1 {} {
                        if iszero(shr(m, xor(mload(subject), s))) {
                            if eq(keccak256(subject, needleLen), h) {
                                result := sub(subject, subjectStart)
                                break
                            }
                        }
                        subject := add(subject, 1)
                        if iszero(lt(subject, end)) { break }
                    }
                    break
                }
                for {} 1 {} {
                    if iszero(shr(m, xor(mload(subject), s))) {
                        result := sub(subject, subjectStart)
                        break
                    }
                    subject := add(subject, 1)
                    if iszero(lt(subject, end)) { break }
                }
                break
            }
        }
    }

    /// @dev Returns the byte index of the first location of `needle` in `subject`,
    /// needleing from left to right, starting from `from`. Optimized for byte needles.
    /// Returns `NOT_FOUND` (i.e. `type(uint256).max`) if the `needle` is not found.
    function indexOfByte(bytes memory subject, bytes1 needle, uint256 from)
        internal
        pure
        returns (uint256 result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            result := not(0) // Initialize to `NOT_FOUND`.
            if gt(mload(subject), from) {
                let start := add(subject, 0x20)
                let end := add(start, mload(subject))
                let m := div(not(0), 255) // `0x0101 ... `.
                let h := mul(byte(0, needle), m) // Replicating needle mask.
                m := not(shl(7, m)) // `0x7f7f ... `.
                for { let i := add(start, from) } 1 {} {
                    let c := xor(mload(i), h) // Load 32-byte chunk and xor with mask.
                    c := not(or(or(add(and(c, m), m), c), m)) // Each needle byte will be `0x80`.
                    if c {
                        c := and(not(shr(shl(3, sub(end, i)), not(0))), c) // Truncate bytes past the end.
                        if c {
                            let r := shl(7, lt(0x8421084210842108cc6318c6db6d54be, c)) // Save bytecode.
                            r := or(shl(6, lt(0xffffffffffffffff, shr(r, c))), r)
                            // forgefmt: disable-next-item
                            result := add(sub(i, start), shr(3, xor(byte(and(0x1f, shr(byte(24,
                                mul(0x02040810204081, shr(r, c))), 0x8421084210842108cc6318c6db6d54be)),
                                0xc0c8c8d0c8e8d0d8c8e8e0e8d0d8e0f0c8d0e8d0e0e0d8f0d0d0e0d8f8f8f8f8), r)))
                            break
                        }
                    }
                    i := add(i, 0x20)
                    if iszero(lt(i, end)) { break }
                }
            }
        }
    }

    /// @dev Returns the byte index of the first location of `needle` in `subject`,
    /// needleing from left to right. Optimized for byte needles.
    /// Returns `NOT_FOUND` (i.e. `type(uint256).max`) if the `needle` is not found.
    function indexOfByte(bytes memory subject, bytes1 needle)
        internal
        pure
        returns (uint256 result)
    {
        return indexOfByte(subject, needle, 0);
    }

    /// @dev Returns the byte index of the first location of `needle` in `subject`,
    /// needleing from left to right.
    /// Returns `NOT_FOUND` (i.e. `type(uint256).max`) if the `needle` is not found.
    function indexOf(bytes memory subject, bytes memory needle) internal pure returns (uint256) {
        return indexOf(subject, needle, 0);
    }

    /// @dev Returns the byte index of the first location of `needle` in `subject`,
    /// needleing from right to left, starting from `from`.
    /// Returns `NOT_FOUND` (i.e. `type(uint256).max`) if the `needle` is not found.
    function lastIndexOf(bytes memory subject, bytes memory needle, uint256 from)
        internal
        pure
        returns (uint256 result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            for {} 1 {} {
                result := not(0) // Initialize to `NOT_FOUND`.
                let needleLen := mload(needle)
                if gt(needleLen, mload(subject)) { break }
                let w := result

                let fromMax := sub(mload(subject), needleLen)
                if iszero(gt(fromMax, from)) { from := fromMax }

                let end := add(add(subject, 0x20), w)
                subject := add(add(subject, 0x20), from)
                if iszero(gt(subject, end)) { break }
                // As this function is not too often used,
                // we shall simply use keccak256 for smaller bytecode size.
                for { let h := keccak256(add(needle, 0x20), needleLen) } 1 {} {
                    if eq(keccak256(subject, needleLen), h) {
                        result := sub(subject, add(end, 1))
                        break
                    }
                    subject := add(subject, w) // `sub(subject, 1)`.
                    if iszero(gt(subject, end)) { break }
                }
                break
            }
        }
    }

    /// @dev Returns the byte index of the first location of `needle` in `subject`,
    /// needleing from right to left.
    /// Returns `NOT_FOUND` (i.e. `type(uint256).max`) if the `needle` is not found.
    function lastIndexOf(bytes memory subject, bytes memory needle)
        internal
        pure
        returns (uint256)
    {
        return lastIndexOf(subject, needle, type(uint256).max);
    }

    /// @dev Returns true if `needle` is found in `subject`, false otherwise.
    function contains(bytes memory subject, bytes memory needle) internal pure returns (bool) {
        return indexOf(subject, needle) != NOT_FOUND;
    }

    /// @dev Returns whether `subject` starts with `needle`.
    function startsWith(bytes memory subject, bytes memory needle)
        internal
        pure
        returns (bool result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let n := mload(needle)
            // Just using keccak256 directly is actually cheaper.
            let t := eq(keccak256(add(subject, 0x20), n), keccak256(add(needle, 0x20), n))
            result := lt(gt(n, mload(subject)), t)
        }
    }

    /// @dev Returns whether `subject` ends with `needle`.
    function endsWith(bytes memory subject, bytes memory needle)
        internal
        pure
        returns (bool result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let n := mload(needle)
            let notInRange := gt(n, mload(subject))
            // `subject + 0x20 + max(subject.length - needle.length, 0)`.
            let t := add(add(subject, 0x20), mul(iszero(notInRange), sub(mload(subject), n)))
            // Just using keccak256 directly is actually cheaper.
            result := gt(eq(keccak256(t, n), keccak256(add(needle, 0x20), n)), notInRange)
        }
    }

    /// @dev Returns `subject` repeated `times`.
    function repeat(bytes memory subject, uint256 times)
        internal
        pure
        returns (bytes memory result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let l := mload(subject) // Subject length.
            if iszero(or(iszero(times), iszero(l))) {
                result := mload(0x40)
                subject := add(subject, 0x20)
                let o := add(result, 0x20)
                for {} 1 {} {
                    // Copy the `subject` one word at a time.
                    for { let j := 0 } 1 {} {
                        mstore(add(o, j), mload(add(subject, j)))
                        j := add(j, 0x20)
                        if iszero(lt(j, l)) { break }
                    }
                    o := add(o, l)
                    times := sub(times, 1)
                    if iszero(times) { break }
                }
                mstore(o, 0) // Zeroize the slot after the bytes.
                mstore(0x40, add(o, 0x20)) // Allocate memory.
                mstore(result, sub(o, add(result, 0x20))) // Store the length.
            }
        }
    }

    /// @dev Returns a copy of `subject` sliced from `start` to `end` (exclusive).
    /// `start` and `end` are byte offsets.
    function slice(bytes memory subject, uint256 start, uint256 end)
        internal
        pure
        returns (bytes memory result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let l := mload(subject) // Subject length.
            if iszero(gt(l, end)) { end := l }
            if iszero(gt(l, start)) { start := l }
            if lt(start, end) {
                result := mload(0x40)
                let n := sub(end, start)
                let i := add(subject, start)
                let w := not(0x1f)
                // Copy the `subject` one word at a time, backwards.
                for { let j := and(add(n, 0x1f), w) } 1 {} {
                    mstore(add(result, j), mload(add(i, j)))
                    j := add(j, w) // `sub(j, 0x20)`.
                    if iszero(j) { break }
                }
                let o := add(add(result, 0x20), n)
                mstore(o, 0) // Zeroize the slot after the bytes.
                mstore(0x40, add(o, 0x20)) // Allocate memory.
                mstore(result, n) // Store the length.
            }
        }
    }

    /// @dev Returns a copy of `subject` sliced from `start` to the end of the bytes.
    /// `start` is a byte offset.
    function slice(bytes memory subject, uint256 start)
        internal
        pure
        returns (bytes memory result)
    {
        result = slice(subject, start, type(uint256).max);
    }

    /// @dev Returns a copy of `subject` sliced from `start` to `end` (exclusive).
    /// `start` and `end` are byte offsets. Faster than Solidity's native slicing.
    function sliceCalldata(bytes calldata subject, uint256 start, uint256 end)
        internal
        pure
        returns (bytes calldata result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            end := xor(end, mul(xor(end, subject.length), lt(subject.length, end)))
            start := xor(start, mul(xor(start, subject.length), lt(subject.length, start)))
            result.offset := add(subject.offset, start)
            result.length := mul(lt(start, end), sub(end, start))
        }
    }

    /// @dev Returns a copy of `subject` sliced from `start` to the end of the bytes.
    /// `start` is a byte offset. Faster than Solidity's native slicing.
    function sliceCalldata(bytes calldata subject, uint256 start)
        internal
        pure
        returns (bytes calldata result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            start := xor(start, mul(xor(start, subject.length), lt(subject.length, start)))
            result.offset := add(subject.offset, start)
            result.length := mul(lt(start, subject.length), sub(subject.length, start))
        }
    }

    /// @dev Reduces the size of `subject` to `n`.
    /// If `n` is greater than the size of `subject`, this will be a no-op.
    function truncate(bytes memory subject, uint256 n)
        internal
        pure
        returns (bytes memory result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            result := subject
            mstore(mul(lt(n, mload(result)), result), n)
        }
    }

    /// @dev Returns a copy of `subject`, with the length reduced to `n`.
    /// If `n` is greater than the size of `subject`, this will be a no-op.
    function truncatedCalldata(bytes calldata subject, uint256 n)
        internal
        pure
        returns (bytes calldata result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            result.offset := subject.offset
            result.length := xor(n, mul(xor(n, subject.length), lt(subject.length, n)))
        }
    }

    /// @dev Returns all the indices of `needle` in `subject`.
    /// The indices are byte offsets.
    function indicesOf(bytes memory subject, bytes memory needle)
        internal
        pure
        returns (uint256[] memory result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let searchLen := mload(needle)
            if iszero(gt(searchLen, mload(subject))) {
                result := mload(0x40)
                let i := add(subject, 0x20)
                let o := add(result, 0x20)
                let subjectSearchEnd := add(sub(add(i, mload(subject)), searchLen), 1)
                let h := 0 // The hash of `needle`.
                if iszero(lt(searchLen, 0x20)) { h := keccak256(add(needle, 0x20), searchLen) }
                let s := mload(add(needle, 0x20))
                for { let m := shl(3, sub(0x20, and(searchLen, 0x1f))) } 1 {} {
                    let t := mload(i)
                    // Whether the first `searchLen % 32` bytes of `subject` and `needle` matches.
                    if iszero(shr(m, xor(t, s))) {
                        if h {
                            if iszero(eq(keccak256(i, searchLen), h)) {
                                i := add(i, 1)
                                if iszero(lt(i, subjectSearchEnd)) { break }
                                continue
                            }
                        }
                        mstore(o, sub(i, add(subject, 0x20))) // Append to `result`.
                        o := add(o, 0x20)
                        i := add(i, searchLen) // Advance `i` by `searchLen`.
                        if searchLen {
                            if iszero(lt(i, subjectSearchEnd)) { break }
                            continue
                        }
                    }
                    i := add(i, 1)
                    if iszero(lt(i, subjectSearchEnd)) { break }
                }
                mstore(result, shr(5, sub(o, add(result, 0x20)))) // Store the length of `result`.
                // Allocate memory for result.
                // We allocate one more word, so this array can be recycled for {split}.
                mstore(0x40, add(o, 0x20))
            }
        }
    }

    /// @dev Returns an arrays of bytess based on the `delimiter` inside of the `subject` bytes.
    function split(bytes memory subject, bytes memory delimiter)
        internal
        pure
        returns (bytes[] memory result)
    {
        uint256[] memory indices = indicesOf(subject, delimiter);
        /// @solidity memory-safe-assembly
        assembly {
            let w := not(0x1f)
            let indexPtr := add(indices, 0x20)
            let indicesEnd := add(indexPtr, shl(5, add(mload(indices), 1)))
            mstore(add(indicesEnd, w), mload(subject))
            mstore(indices, add(mload(indices), 1))
            for { let prevIndex := 0 } 1 {} {
                let index := mload(indexPtr)
                mstore(indexPtr, 0x60)
                if iszero(eq(index, prevIndex)) {
                    let element := mload(0x40)
                    let l := sub(index, prevIndex)
                    mstore(element, l) // Store the length of the element.
                    // Copy the `subject` one word at a time, backwards.
                    for { let o := and(add(l, 0x1f), w) } 1 {} {
                        mstore(add(element, o), mload(add(add(subject, prevIndex), o)))
                        o := add(o, w) // `sub(o, 0x20)`.
                        if iszero(o) { break }
                    }
                    mstore(add(add(element, 0x20), l), 0) // Zeroize the slot after the bytes.
                    // Allocate memory for the length and the bytes, rounded up to a multiple of 32.
                    mstore(0x40, add(element, and(add(l, 0x3f), w)))
                    mstore(indexPtr, element) // Store the `element` into the array.
                }
                prevIndex := add(index, mload(delimiter))
                indexPtr := add(indexPtr, 0x20)
                if iszero(lt(indexPtr, indicesEnd)) { break }
            }
            result := indices
            if iszero(mload(delimiter)) {
                result := add(indices, 0x20)
                mstore(result, sub(mload(indices), 2))
            }
        }
    }

    /// @dev Returns a concatenated bytes of `a` and `b`.
    /// Cheaper than `bytes.concat()` and does not de-align the free memory pointer.
    function concat(bytes memory a, bytes memory b) internal pure returns (bytes memory result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := mload(0x40)
            let w := not(0x1f)
            let aLen := mload(a)
            // Copy `a` one word at a time, backwards.
            for { let o := and(add(aLen, 0x20), w) } 1 {} {
                mstore(add(result, o), mload(add(a, o)))
                o := add(o, w) // `sub(o, 0x20)`.
                if iszero(o) { break }
            }
            let bLen := mload(b)
            let output := add(result, aLen)
            // Copy `b` one word at a time, backwards.
            for { let o := and(add(bLen, 0x20), w) } 1 {} {
                mstore(add(output, o), mload(add(b, o)))
                o := add(o, w) // `sub(o, 0x20)`.
                if iszero(o) { break }
            }
            let totalLen := add(aLen, bLen)
            let last := add(add(result, 0x20), totalLen)
            mstore(last, 0) // Zeroize the slot after the bytes.
            mstore(result, totalLen) // Store the length.
            mstore(0x40, add(last, 0x20)) // Allocate memory.
        }
    }

    /// @dev Returns whether `a` equals `b`.
    function eq(bytes memory a, bytes memory b) internal pure returns (bool result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := eq(keccak256(add(a, 0x20), mload(a)), keccak256(add(b, 0x20), mload(b)))
        }
    }

    /// @dev Returns whether `a` equals `b`, where `b` is a null-terminated small bytes.
    function eqs(bytes memory a, bytes32 b) internal pure returns (bool result) {
        /// @solidity memory-safe-assembly
        assembly {
            // These should be evaluated on compile time, as far as possible.
            let m := not(shl(7, div(not(iszero(b)), 255))) // `0x7f7f ...`.
            let x := not(or(m, or(b, add(m, and(b, m)))))
            let r := shl(7, iszero(iszero(shr(128, x))))
            r := or(r, shl(6, iszero(iszero(shr(64, shr(r, x))))))
            r := or(r, shl(5, lt(0xffffffff, shr(r, x))))
            r := or(r, shl(4, lt(0xffff, shr(r, x))))
            r := or(r, shl(3, lt(0xff, shr(r, x))))
            // forgefmt: disable-next-item
            result := gt(eq(mload(a), add(iszero(x), xor(31, shr(3, r)))),
                xor(shr(add(8, r), b), shr(add(8, r), mload(add(a, 0x20)))))
        }
    }

    /// @dev Returns 0 if `a == b`, -1 if `a < b`, +1 if `a > b`.
    /// If `a` == b[:a.length]`, and `a.length < b.length`, returns -1.
    function cmp(bytes memory a, bytes memory b) internal pure returns (int256 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let aLen := mload(a)
            let bLen := mload(b)
            let n := and(xor(aLen, mul(xor(aLen, bLen), lt(bLen, aLen))), not(0x1f))
            if n {
                for { let i := 0x20 } 1 {} {
                    let x := mload(add(a, i))
                    let y := mload(add(b, i))
                    if iszero(or(xor(x, y), eq(i, n))) {
                        i := add(i, 0x20)
                        continue
                    }
                    result := sub(gt(x, y), lt(x, y))
                    break
                }
            }
            // forgefmt: disable-next-item
            if iszero(result) {
                let l := 0x201f1e1d1c1b1a191817161514131211100f0e0d0c0b0a090807060504030201
                let x := and(mload(add(add(a, 0x20), n)), shl(shl(3, byte(sub(aLen, n), l)), not(0)))
                let y := and(mload(add(add(b, 0x20), n)), shl(shl(3, byte(sub(bLen, n), l)), not(0)))
                result := sub(gt(x, y), lt(x, y))
                if iszero(result) { result := sub(gt(aLen, bLen), lt(aLen, bLen)) }
            }
        }
    }

    /// @dev Directly returns `a` without copying.
    function directReturn(bytes memory a) internal pure {
        /// @solidity memory-safe-assembly
        assembly {
            // Assumes that the bytes does not start from the scratch space.
            let retStart := sub(a, 0x20)
            let retUnpaddedSize := add(mload(a), 0x40)
            // Right pad with zeroes. Just in case the bytes is produced
            // by a method that doesn't zero right pad.
            mstore(add(retStart, retUnpaddedSize), 0)
            mstore(retStart, 0x20) // Store the return offset.
            // End the transaction, returning the bytes.
            return(retStart, and(not(0x1f), add(0x1f, retUnpaddedSize)))
        }
    }

    /// @dev Directly returns `a` with minimal copying.
    function directReturn(bytes[] memory a) internal pure {
        /// @solidity memory-safe-assembly
        assembly {
            let n := mload(a) // `a.length`.
            let o := add(a, 0x20) // Start of elements in `a`.
            let u := a // Highest memory slot.
            let w := not(0x1f)
            for { let i := 0 } iszero(eq(i, n)) { i := add(i, 1) } {
                let c := add(o, shl(5, i)) // Location of pointer to `a[i]`.
                let s := mload(c) // `a[i]`.
                let l := mload(s) // `a[i].length`.
                let r := and(l, 0x1f) // `a[i].length % 32`.
                let z := add(0x20, and(l, w)) // Offset of last word in `a[i]` from `s`.
                // If `s` comes before `o`, or `s` is not zero right padded.
                if iszero(lt(lt(s, o), or(iszero(r), iszero(shl(shl(3, r), mload(add(s, z))))))) {
                    let m := mload(0x40)
                    mstore(m, l) // Copy `a[i].length`.
                    for {} 1 {} {
                        mstore(add(m, z), mload(add(s, z))) // Copy `a[i]`, backwards.
                        z := add(z, w) // `sub(z, 0x20)`.
                        if iszero(z) { break }
                    }
                    let e := add(add(m, 0x20), l)
                    mstore(e, 0) // Zeroize the slot after the copied bytes.
                    mstore(0x40, add(e, 0x20)) // Allocate memory.
                    s := m
                }
                mstore(c, sub(s, o)) // Convert to calldata offset.
                let t := add(l, add(s, 0x20))
                if iszero(lt(t, u)) { u := t }
            }
            let retStart := add(a, w) // Assumes `a` doesn't start from scratch space.
            mstore(retStart, 0x20) // Store the return offset.
            return(retStart, add(0x40, sub(u, retStart))) // End the transaction.
        }
    }

    /// @dev Returns the word at `offset`, without any bounds checks.
    function load(bytes memory a, uint256 offset) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := mload(add(add(a, 0x20), offset))
        }
    }

    /// @dev Returns the word at `offset`, without any bounds checks.
    function loadCalldata(bytes calldata a, uint256 offset)
        internal
        pure
        returns (bytes32 result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            result := calldataload(add(a.offset, offset))
        }
    }

    /// @dev Returns a slice representing a static struct in the calldata. Performs bounds checks.
    function staticStructInCalldata(bytes calldata a, uint256 offset)
        internal
        pure
        returns (bytes calldata result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let l := sub(a.length, 0x20)
            result.offset := add(a.offset, offset)
            result.length := sub(a.length, offset)
            if or(shr(64, or(l, a.offset)), gt(offset, l)) { revert(l, 0x00) }
        }
    }

    /// @dev Returns a slice representing a dynamic struct in the calldata. Performs bounds checks.
    function dynamicStructInCalldata(bytes calldata a, uint256 offset)
        internal
        pure
        returns (bytes calldata result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let l := sub(a.length, 0x20)
            let s := calldataload(add(a.offset, offset)) // Relative offset of `result` from `a.offset`.
            result.offset := add(a.offset, s)
            result.length := sub(a.length, s)
            if or(shr(64, or(s, or(l, a.offset))), gt(offset, l)) { revert(l, 0x00) }
        }
    }

    /// @dev Returns bytes in calldata. Performs bounds checks.
    function bytesInCalldata(bytes calldata a, uint256 offset)
        internal
        pure
        returns (bytes calldata result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let l := sub(a.length, 0x20)
            let s := calldataload(add(a.offset, offset)) // Relative offset of `result` from `a.offset`.
            result.offset := add(add(a.offset, s), 0x20)
            result.length := calldataload(add(a.offset, s))
            // forgefmt: disable-next-item
            if or(shr(64, or(result.length, or(s, or(l, a.offset)))),
                or(gt(add(s, result.length), l), gt(offset, l))) { revert(l, 0x00) }
        }
    }

    /// @dev Checks if `x` is in `a`. Assumes `a` has been checked.
    function checkInCalldata(bytes calldata x, bytes calldata a) internal pure {
        /// @solidity memory-safe-assembly
        assembly {
            if or(
                or(lt(x.offset, a.offset), gt(add(x.offset, x.length), add(a.length, a.offset))),
                shr(64, or(x.length, x.offset))
            ) { revert(0x00, 0x00) }
        }
    }

    /// @dev Checks if `x` is in `a`. Assumes `a` has been checked.
    function checkInCalldata(bytes[] calldata x, bytes calldata a) internal pure {
        /// @solidity memory-safe-assembly
        assembly {
            let e := sub(add(a.length, a.offset), 0x20)
            if or(lt(x.offset, a.offset), shr(64, x.offset)) { revert(0x00, 0x00) }
            for { let i := 0 } iszero(eq(x.length, i)) { i := add(i, 1) } {
                let o := calldataload(add(x.offset, shl(5, i)))
                let t := add(o, x.offset)
                let l := calldataload(t)
                if or(shr(64, or(l, o)), gt(add(t, l), e)) { revert(0x00, 0x00) }
            }
        }
    }

    /// @dev Returns empty calldata bytes. For silencing the compiler.
    function emptyCalldata() internal pure returns (bytes calldata result) {
        /// @solidity memory-safe-assembly
        assembly {
            result.length := 0
        }
    }

    /// @dev Returns the most significant 20 bytes as an address.
    function msbToAddress(bytes32 x) internal pure returns (address) {
        return address(bytes20(x));
    }

    /// @dev Returns the least significant 20 bytes as an address.
    function lsbToAddress(bytes32 x) internal pure returns (address) {
        return address(uint160(uint256(x)));
    }
}
