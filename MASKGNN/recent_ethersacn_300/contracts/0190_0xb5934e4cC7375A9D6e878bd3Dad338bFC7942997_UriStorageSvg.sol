// ===== FILE: src/nft/UriStorageSvg.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Initializable} from "@solady/src/utils/Initializable.sol";
import {Ownable} from "@solady/src/auth/Ownable.sol";
import {LibString} from "@solady/src/utils/LibString.sol";
import {LibString} from "@solady/src/utils/LibString.sol";
import {LibBytes} from "@solady/src/utils/LibBytes.sol";
import {Base64} from "@solady/src/utils/Base64.sol";
import {EfficientHashLib} from "@solady/src/utils/EfficientHashLib.sol";

import {IHaNft} from "./IHaNft.sol";
import {IUriStorage} from "./IUriStorage.sol";

contract UriStorageSvg is Initializable, Ownable, IUriStorage {
    using LibString for uint256;
    using LibBytes for bytes;

    /// @dev Indicates that the caller is not the associated NFT contract
    error MsgSenderNotNft(address sender, address nft);

    /// @dev Struct to store an attribute of the NFT
    struct Attribute {
        string name;
        uint256 traitsIndex;
    }

    /// @dev Struct to store a trait of the NFT
    struct Trait {
        string name;
        bytes[] svgElements;
        uint16 chance;
    }

    /// @dev The description of the NFT
    string private _description;

    /// @dev The SVG prefix of the NFT
    string private _svgPrefix;

    /// @dev The colors of the NFT
    uint24[] private _colors;

    /// @dev The attributes of the NFT
    Attribute[] private _attributes;
    
    /// @dev The traits of the NFT
    mapping(uint256 attributeIndex => Trait[]) private _traits;

    /// @dev Mapping of SVG hashes to their used status
    mapping(bytes32 svgHash => bool) private _usedSvgHashes;

    /// @dev Mapping of token IDs to their SVG hashes
    mapping(uint256 tokenId => bytes32) private _svgHashes;

    /// @dev Nonce used for new token IDs
    uint256 private _nonce;

    /// @dev The address of the associated NFT contract
    address public nft;

    constructor() {}

    /**
     * @dev Modifier to check if the caller is the associated NFT contract
     */
    modifier onlyNft() {
        _onlyNft();
        _;
    }

    /// @inheritdoc IUriStorage
    function tokenURI(
        uint256 tokenId
    ) external view override returns (string memory) {
        string memory svg = _generateSvg(_svgHashes[tokenId]);
        string memory svgData = string.concat(
            "data:image/svg+xml;base64,",
            Base64.encode(bytes(svg))
        );

        string memory attributes = _generateAttributes(_svgHashes[tokenId]);

        string memory metadata = string.concat(
            '{"name":"',
            IHaNft(nft).name(),
            " #",
            tokenId.toString(),
            '","description":"',
            _description,
            '","image":"',
            svgData,
            '","attributes":',
            attributes,
            "}"
        );

        return string.concat(
            "data:application/json;base64,",
            Base64.encode(bytes(metadata))
        );
    }

    /**
     * @notice Initializes the URI storage contract
     * @param owner_ The address of the owner of the contract
     */
    function initialize(address owner_) external initializer {
        _initializeOwner(owner_);
    }

    /**
     * @notice Sets the description of the NFT
     * @param description The description of the NFT
     */
    function setDescription(string memory description) external onlyOwner {
        _description = description;
    }

    /**
     * @notice Sets the SVG prefix of the NFT
     * @param svgPrefix The SVG prefix of the NFT
     */
    function setSvgPrefix(string memory svgPrefix) external onlyOwner {
        _svgPrefix = svgPrefix;
    }

    /**
     * @notice Adds colors to the NFT
     * @param colors The colors to add to the NFT
     */
    function addColors(uint24[] memory colors) external onlyOwner {
        for (uint256 i = 0; i < colors.length; i++) {
            _colors.push(colors[i]);
        }
    }

    /**
     * @notice Adds attributes to the NFT
     * @param attributes The attributes to add to the NFT
     */
    function addAttributes(Attribute[] memory attributes) external onlyOwner {
        for (uint256 i = 0; i < attributes.length; i++) {
            _attributes.push(attributes[i]);
        }
    }

    /**
     * @notice Adds traits to the NFT
     * @param indexes The indexes of the attributes to add the traits to
     * @param traits The traits to add to the NFT
     */
    function addTraits(uint256[] memory indexes, Trait[][] memory traits) external onlyOwner {
        for (uint256 i = 0; i < indexes.length; i++) {
            for (uint256 j = 0; j < traits[i].length; j++) {
                _traits[indexes[i]].push(traits[i][j]);
            }
        }
    }

    /**
     * @notice Sets the NFT contract
     * @dev This function renounces ownership of the contract
     * @param nft_ The address of the NFT contract
     */
    function setNft(address nft_) external onlyOwner {
        nft = nft_;

        _setOwner(address(0));
    }

    /// @inheritdoc IUriStorage
    function drawTokenId(
        bytes32 random, 
        uint256 /* totalSupply */,
        uint256 /* maxSupply */
    ) external override onlyNft returns (uint256 tokenId) {
        bytes32 svgHash = _generateSvgHash(random);

        if(_usedSvgHashes[svgHash]) {
            return type(uint256).max;
        }

        tokenId = _nonce + 1;
        _nonce++;

        _usedSvgHashes[svgHash] = true;
        _svgHashes[tokenId] = svgHash;
    }

    /// @inheritdoc IUriStorage
    function returnTokenId(
        uint256 tokenId, 
        uint256 /* totalSupply */,
        uint256 /* maxSupply */
    ) external override onlyNft {
        delete _usedSvgHashes[_svgHashes[tokenId]];
        delete _svgHashes[tokenId];
    }

    /**
     * @dev Internal function to generate an SVG hash
     * @param random The random number
     * @return svgHash The SVG hash
     */
    function _generateSvgHash(bytes32 random) internal view returns (bytes32 svgHash) {
        for (uint256 i = 0; i < _attributes.length; i++) {
            if(i == 16) {
                random = EfficientHashLib.hash(random);
            }
            uint256 idx = i < 16 ? i : i - 16;

            uint16 roll = (uint16(uint8(random[idx * 2])) << 8) | uint16(uint8(random[idx * 2 + 1]));
            
            Trait[] memory traits = _traits[_attributes[i].traitsIndex];
            uint256 traitIndex = type(uint8).max;
            uint256 cumulativeWeight;
            for(uint256 j = 0; j < traits.length; j++) {
                cumulativeWeight += traits[j].chance;
                if (roll < cumulativeWeight) {
                    traitIndex = j;
                    break;
                }
            }

            svgHash |= bytes32(uint256(uint8(traitIndex))) << (8 * (31 - i));
        }
    }

    /**
     * @dev Internal function to generate an SVG
     * @param svgHash The SVG hash
     * @return svg The SVG
     */
    function _generateSvg(bytes32 svgHash) internal view returns (string memory svg) {
        svg = _svgPrefix;

        for(uint256 i = 0; i < _attributes.length; i++) {
            uint256 traitIndex = uint8(svgHash[i]);
            if(traitIndex == type(uint8).max) {
                continue;
            }

            bytes[] memory elements = _traits[_attributes[i].traitsIndex][traitIndex].svgElements;
            for(uint256 j = 0; j < elements.length; j++) {
                bytes memory element = elements[j];
                uint8 selector = uint8(element[0]);

                if (selector == 0) {
                    // 0: Custom SVG element
                    svg = string.concat(
                        svg,
                        string(element.slice(1))
                    );
                } else if (selector == 1) {
                    uint256 x = (uint16(uint8(element[1])) << 8) | uint16(uint8(element[2]));
                    uint256 y = (uint16(uint8(element[3])) << 8) | uint16(uint8(element[4]));
                    uint256 width = (uint16(uint8(element[5])) << 8) | uint16(uint8(element[6]));
                    uint256 height = (uint16(uint8(element[7])) << 8) | uint16(uint8(element[8]));
                    uint256 color = _colors[(uint16(uint8(element[9])) << 8) | uint16(uint8(element[10]))];

                    svg = string.concat(
                        svg,
                        '<rect x="',
                        x.toString(),
                        '" y="',
                        y.toString(),
                        '" width="',
                        width.toString(),
                        '" height="',
                        height.toString(),
                        '" fill="#',
                        color.toHexStringNoPrefix(3),
                        '"/>'
                    );
                }
            }
        }
    }

    /**
     * @dev Internal function to generate attributes
     * @param svgHash The SVG hash
     * @return attributes The attributes
     */
    function _generateAttributes(bytes32 svgHash) internal view returns (string memory attributes) {
        uint256 attributesCount = 0;
        for (uint256 i = 0; i < _attributes.length; i++) {
            uint256 traitIndex = uint8(svgHash[i]);
            if(traitIndex == type(uint8).max) {
                continue;
            }

            attributes = string.concat(
                attributes,
                attributesCount > 0 ? "," : "",
                '{"trait_type":"',
                _attributes[i].name,
                '","value":"',
                _traits[_attributes[i].traitsIndex][traitIndex].name,
                '"}'
            );
            attributesCount++;
        }

        attributes = string.concat("[", attributes, "]");
    }

    /**
     * @dev Internal function to check if the caller is the associated NFT contract
     */
    function _onlyNft() internal view {
        if (msg.sender != nft) {
            revert MsgSenderNotNft(msg.sender, nft);
        }
    }
}

// ===== FILE: lib/solady/src/utils/Initializable.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

/// @notice Initializable mixin for the upgradeable contracts.
/// @author Solady (https://github.com/vectorized/solady/blob/main/src/utils/Initializable.sol)
/// @author Modified from OpenZeppelin (https://github.com/OpenZeppelin/openzeppelin-contracts/tree/master/contracts/proxy/utils/Initializable.sol)
abstract contract Initializable {
    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                       CUSTOM ERRORS                        */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev The contract is already initialized.
    error InvalidInitialization();

    /// @dev The contract is not initializing.
    error NotInitializing();

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                           EVENTS                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Triggered when the contract has been initialized.
    event Initialized(uint64 version);

    /// @dev `keccak256(bytes("Initialized(uint64)"))`.
    bytes32 private constant _INITIALIZED_EVENT_SIGNATURE =
        0xc7f505b2f371ae2175ee4913f4499e1f2633a7b5936321eed1cdaeb6115181d2;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                          STORAGE                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev The default initializable slot is given by:
    /// `bytes32(~uint256(uint32(bytes4(keccak256("_INITIALIZABLE_SLOT")))))`.
    ///
    /// Bits Layout:
    /// - [0]     `initializing`
    /// - [1..64] `initializedVersion`
    bytes32 private constant _INITIALIZABLE_SLOT =
        0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffbf601132;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                        CONSTRUCTOR                         */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    constructor() {
        // Construction time check to ensure that `_initializableSlot()` is not
        // overridden to zero. Will be optimized away if there is no revert.
        require(_initializableSlot() != bytes32(0));
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                         OPERATIONS                         */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Override to return a non-zero custom storage slot if required.
    function _initializableSlot() internal pure virtual returns (bytes32) {
        return _INITIALIZABLE_SLOT;
    }

    /// @dev Guards an initializer function so that it can be invoked at most once.
    ///
    /// You can guard a function with `onlyInitializing` such that it can be called
    /// through a function guarded with `initializer`.
    ///
    /// This is similar to `reinitializer(1)`, except that in the context of a constructor,
    /// an `initializer` guarded function can be invoked multiple times.
    /// This can be useful during testing and is not expected to be used in production.
    ///
    /// Emits an {Initialized} event.
    modifier initializer() virtual {
        bytes32 s = _initializableSlot();
        /// @solidity memory-safe-assembly
        assembly {
            let i := sload(s)
            // Set `initializing` to 1, `initializedVersion` to 1.
            sstore(s, 3)
            // If `!(initializing == 0 && initializedVersion == 0)`.
            if i {
                // If `!(address(this).code.length == 0 && initializedVersion == 1)`.
                if iszero(lt(extcodesize(address()), eq(shr(1, i), 1))) {
                    mstore(0x00, 0xf92ee8a9) // `InvalidInitialization()`.
                    revert(0x1c, 0x04)
                }
                s := shl(shl(255, i), s) // Skip initializing if `initializing == 1`.
            }
        }
        _;
        /// @solidity memory-safe-assembly
        assembly {
            if s {
                // Set `initializing` to 0, `initializedVersion` to 1.
                sstore(s, 2)
                // Emit the {Initialized} event.
                mstore(0x20, 1)
                log1(0x20, 0x20, _INITIALIZED_EVENT_SIGNATURE)
            }
        }
    }

    /// @dev Guards a reinitializer function so that it can be invoked at most once.
    ///
    /// You can guard a function with `onlyInitializing` such that it can be called
    /// through a function guarded with `reinitializer`.
    ///
    /// Emits an {Initialized} event.
    modifier reinitializer(uint64 version) virtual {
        bytes32 s = _initializableSlot();
        /// @solidity memory-safe-assembly
        assembly {
            // Clean upper bits, and shift left by 1 to make space for the initializing bit.
            version := shl(1, and(version, 0xffffffffffffffff))
            let i := sload(s)
            // If `initializing == 1 || initializedVersion >= version`.
            if iszero(lt(and(i, 1), lt(i, version))) {
                mstore(0x00, 0xf92ee8a9) // `InvalidInitialization()`.
                revert(0x1c, 0x04)
            }
            // Set `initializing` to 1, `initializedVersion` to `version`.
            sstore(s, or(1, version))
        }
        _;
        /// @solidity memory-safe-assembly
        assembly {
            // Set `initializing` to 0, `initializedVersion` to `version`.
            sstore(s, version)
            // Emit the {Initialized} event.
            mstore(0x20, shr(1, version))
            log1(0x20, 0x20, _INITIALIZED_EVENT_SIGNATURE)
        }
    }

    /// @dev Guards a function such that it can only be called in the scope
    /// of a function guarded with `initializer` or `reinitializer`.
    modifier onlyInitializing() virtual {
        _checkInitializing();
        _;
    }

    /// @dev Reverts if the contract is not initializing.
    function _checkInitializing() internal view virtual {
        bytes32 s = _initializableSlot();
        /// @solidity memory-safe-assembly
        assembly {
            if iszero(and(1, sload(s))) {
                mstore(0x00, 0xd7e6bcf8) // `NotInitializing()`.
                revert(0x1c, 0x04)
            }
        }
    }

    /// @dev Locks any future initializations by setting the initialized version to `2**64 - 1`.
    ///
    /// Calling this in the constructor will prevent the contract from being initialized
    /// or reinitialized. It is recommended to use this to lock implementation contracts
    /// that are designed to be called through proxies.
    ///
    /// Emits an {Initialized} event the first time it is successfully called.
    function _disableInitializers() internal virtual {
        bytes32 s = _initializableSlot();
        /// @solidity memory-safe-assembly
        assembly {
            let i := sload(s)
            if and(i, 1) {
                mstore(0x00, 0xf92ee8a9) // `InvalidInitialization()`.
                revert(0x1c, 0x04)
            }
            let uint64max := 0xffffffffffffffff
            if iszero(eq(shr(1, i), uint64max)) {
                // Set `initializing` to 0, `initializedVersion` to `2**64 - 1`.
                sstore(s, shl(1, uint64max))
                // Emit the {Initialized} event.
                mstore(0x20, uint64max)
                log1(0x20, 0x20, _INITIALIZED_EVENT_SIGNATURE)
            }
        }
    }

    /// @dev Returns the highest version that has been initialized.
    function _getInitializedVersion() internal view virtual returns (uint64 version) {
        bytes32 s = _initializableSlot();
        /// @solidity memory-safe-assembly
        assembly {
            version := shr(1, sload(s))
        }
    }

    /// @dev Returns whether the contract is currently initializing.
    function _isInitializing() internal view virtual returns (bool result) {
        bytes32 s = _initializableSlot();
        /// @solidity memory-safe-assembly
        assembly {
            result := and(1, sload(s))
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
            for {} // Scan for '\0'.
             byte(n, s) { n := add(n, 1) } {}
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
            for {} // Scan for '\0'.
             byte(result, s) { result := add(result, 1) } {}
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
            result := mul(
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
            result := mul(
                or( // Load the length and the bytes of `a` and `b`.
                    shl(shl(3, sub(0x1f, aLen)), mload(add(a, aLen))),
                    shr(shl(3, add(aLen, 1)), mload(add(b, 0x1f)))
                ),
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
                if iszero(lt(needleLen, 0x20)) {
                    h := keccak256(add(needle, 0x20), needleLen)
                }
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
    function truncate(bytes memory subject, uint256 n) internal pure returns (bytes memory result) {
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
                if iszero(lt(searchLen, 0x20)) {
                    h := keccak256(add(needle, 0x20), searchLen)
                }
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
    function loadCalldata(bytes calldata a, uint256 offset) internal pure returns (bytes32 result) {
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
    function encode(bytes memory data, bool fileSafe) internal pure returns (string memory result) {
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


// ===== FILE: lib/solady/src/utils/EfficientHashLib.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

/// @notice Library for efficiently performing keccak256 hashes.
/// @author Solady (https://github.com/vectorized/solady/blob/main/src/utils/EfficientHashLib.sol)
/// @dev To avoid stack-too-deep, you can use:
/// ```
/// bytes32[] memory buffer = EfficientHashLib.malloc(10);
/// EfficientHashLib.set(buffer, 0, value0);
/// ..
/// EfficientHashLib.set(buffer, 9, value9);
/// bytes32 finalHash = EfficientHashLib.hash(buffer);
/// ```
library EfficientHashLib {
    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*               MALLOC-LESS HASHING OPERATIONS               */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Returns `keccak256(abi.encode(v0))`.
    function hash(bytes32 v0) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, v0)
            result := keccak256(0x00, 0x20)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0))`.
    function hash(uint256 v0) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, v0)
            result := keccak256(0x00, 0x20)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, v1))`.
    function hash(bytes32 v0, bytes32 v1) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, v0)
            mstore(0x20, v1)
            result := keccak256(0x00, 0x40)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, v1))`.
    function hash(uint256 v0, uint256 v1) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, v0)
            mstore(0x20, v1)
            result := keccak256(0x00, 0x40)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, v1, v2))`.
    function hash(bytes32 v0, bytes32 v1, bytes32 v2) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            result := keccak256(m, 0x60)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, v1, v2))`.
    function hash(uint256 v0, uint256 v1, uint256 v2) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            result := keccak256(m, 0x60)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, v1, v2, v3))`.
    function hash(bytes32 v0, bytes32 v1, bytes32 v2, bytes32 v3)
        internal
        pure
        returns (bytes32 result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            result := keccak256(m, 0x80)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, v1, v2, v3))`.
    function hash(uint256 v0, uint256 v1, uint256 v2, uint256 v3)
        internal
        pure
        returns (bytes32 result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            result := keccak256(m, 0x80)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v4))`.
    function hash(bytes32 v0, bytes32 v1, bytes32 v2, bytes32 v3, bytes32 v4)
        internal
        pure
        returns (bytes32 result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            result := keccak256(m, 0xa0)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v4))`.
    function hash(uint256 v0, uint256 v1, uint256 v2, uint256 v3, uint256 v4)
        internal
        pure
        returns (bytes32 result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            result := keccak256(m, 0xa0)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v5))`.
    function hash(bytes32 v0, bytes32 v1, bytes32 v2, bytes32 v3, bytes32 v4, bytes32 v5)
        internal
        pure
        returns (bytes32 result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            result := keccak256(m, 0xc0)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v5))`.
    function hash(uint256 v0, uint256 v1, uint256 v2, uint256 v3, uint256 v4, uint256 v5)
        internal
        pure
        returns (bytes32 result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            result := keccak256(m, 0xc0)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v6))`.
    function hash(
        bytes32 v0,
        bytes32 v1,
        bytes32 v2,
        bytes32 v3,
        bytes32 v4,
        bytes32 v5,
        bytes32 v6
    ) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            mstore(add(m, 0xc0), v6)
            result := keccak256(m, 0xe0)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v6))`.
    function hash(
        uint256 v0,
        uint256 v1,
        uint256 v2,
        uint256 v3,
        uint256 v4,
        uint256 v5,
        uint256 v6
    ) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            mstore(add(m, 0xc0), v6)
            result := keccak256(m, 0xe0)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v7))`.
    function hash(
        bytes32 v0,
        bytes32 v1,
        bytes32 v2,
        bytes32 v3,
        bytes32 v4,
        bytes32 v5,
        bytes32 v6,
        bytes32 v7
    ) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            mstore(add(m, 0xc0), v6)
            mstore(add(m, 0xe0), v7)
            result := keccak256(m, 0x100)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v7))`.
    function hash(
        uint256 v0,
        uint256 v1,
        uint256 v2,
        uint256 v3,
        uint256 v4,
        uint256 v5,
        uint256 v6,
        uint256 v7
    ) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            mstore(add(m, 0xc0), v6)
            mstore(add(m, 0xe0), v7)
            result := keccak256(m, 0x100)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v8))`.
    function hash(
        bytes32 v0,
        bytes32 v1,
        bytes32 v2,
        bytes32 v3,
        bytes32 v4,
        bytes32 v5,
        bytes32 v6,
        bytes32 v7,
        bytes32 v8
    ) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            mstore(add(m, 0xc0), v6)
            mstore(add(m, 0xe0), v7)
            mstore(add(m, 0x100), v8)
            result := keccak256(m, 0x120)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v8))`.
    function hash(
        uint256 v0,
        uint256 v1,
        uint256 v2,
        uint256 v3,
        uint256 v4,
        uint256 v5,
        uint256 v6,
        uint256 v7,
        uint256 v8
    ) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            mstore(add(m, 0xc0), v6)
            mstore(add(m, 0xe0), v7)
            mstore(add(m, 0x100), v8)
            result := keccak256(m, 0x120)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v9))`.
    function hash(
        bytes32 v0,
        bytes32 v1,
        bytes32 v2,
        bytes32 v3,
        bytes32 v4,
        bytes32 v5,
        bytes32 v6,
        bytes32 v7,
        bytes32 v8,
        bytes32 v9
    ) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            mstore(add(m, 0xc0), v6)
            mstore(add(m, 0xe0), v7)
            mstore(add(m, 0x100), v8)
            mstore(add(m, 0x120), v9)
            result := keccak256(m, 0x140)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v9))`.
    function hash(
        uint256 v0,
        uint256 v1,
        uint256 v2,
        uint256 v3,
        uint256 v4,
        uint256 v5,
        uint256 v6,
        uint256 v7,
        uint256 v8,
        uint256 v9
    ) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            mstore(add(m, 0xc0), v6)
            mstore(add(m, 0xe0), v7)
            mstore(add(m, 0x100), v8)
            mstore(add(m, 0x120), v9)
            result := keccak256(m, 0x140)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v10))`.
    function hash(
        bytes32 v0,
        bytes32 v1,
        bytes32 v2,
        bytes32 v3,
        bytes32 v4,
        bytes32 v5,
        bytes32 v6,
        bytes32 v7,
        bytes32 v8,
        bytes32 v9,
        bytes32 v10
    ) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            mstore(add(m, 0xc0), v6)
            mstore(add(m, 0xe0), v7)
            mstore(add(m, 0x100), v8)
            mstore(add(m, 0x120), v9)
            mstore(add(m, 0x140), v10)
            result := keccak256(m, 0x160)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v10))`.
    function hash(
        uint256 v0,
        uint256 v1,
        uint256 v2,
        uint256 v3,
        uint256 v4,
        uint256 v5,
        uint256 v6,
        uint256 v7,
        uint256 v8,
        uint256 v9,
        uint256 v10
    ) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            mstore(add(m, 0xc0), v6)
            mstore(add(m, 0xe0), v7)
            mstore(add(m, 0x100), v8)
            mstore(add(m, 0x120), v9)
            mstore(add(m, 0x140), v10)
            result := keccak256(m, 0x160)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v11))`.
    function hash(
        bytes32 v0,
        bytes32 v1,
        bytes32 v2,
        bytes32 v3,
        bytes32 v4,
        bytes32 v5,
        bytes32 v6,
        bytes32 v7,
        bytes32 v8,
        bytes32 v9,
        bytes32 v10,
        bytes32 v11
    ) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            mstore(add(m, 0xc0), v6)
            mstore(add(m, 0xe0), v7)
            mstore(add(m, 0x100), v8)
            mstore(add(m, 0x120), v9)
            mstore(add(m, 0x140), v10)
            mstore(add(m, 0x160), v11)
            result := keccak256(m, 0x180)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v11))`.
    function hash(
        uint256 v0,
        uint256 v1,
        uint256 v2,
        uint256 v3,
        uint256 v4,
        uint256 v5,
        uint256 v6,
        uint256 v7,
        uint256 v8,
        uint256 v9,
        uint256 v10,
        uint256 v11
    ) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            mstore(add(m, 0xc0), v6)
            mstore(add(m, 0xe0), v7)
            mstore(add(m, 0x100), v8)
            mstore(add(m, 0x120), v9)
            mstore(add(m, 0x140), v10)
            mstore(add(m, 0x160), v11)
            result := keccak256(m, 0x180)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v12))`.
    function hash(
        bytes32 v0,
        bytes32 v1,
        bytes32 v2,
        bytes32 v3,
        bytes32 v4,
        bytes32 v5,
        bytes32 v6,
        bytes32 v7,
        bytes32 v8,
        bytes32 v9,
        bytes32 v10,
        bytes32 v11,
        bytes32 v12
    ) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            mstore(add(m, 0xc0), v6)
            mstore(add(m, 0xe0), v7)
            mstore(add(m, 0x100), v8)
            mstore(add(m, 0x120), v9)
            mstore(add(m, 0x140), v10)
            mstore(add(m, 0x160), v11)
            mstore(add(m, 0x180), v12)
            result := keccak256(m, 0x1a0)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v12))`.
    function hash(
        uint256 v0,
        uint256 v1,
        uint256 v2,
        uint256 v3,
        uint256 v4,
        uint256 v5,
        uint256 v6,
        uint256 v7,
        uint256 v8,
        uint256 v9,
        uint256 v10,
        uint256 v11,
        uint256 v12
    ) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            mstore(add(m, 0xc0), v6)
            mstore(add(m, 0xe0), v7)
            mstore(add(m, 0x100), v8)
            mstore(add(m, 0x120), v9)
            mstore(add(m, 0x140), v10)
            mstore(add(m, 0x160), v11)
            mstore(add(m, 0x180), v12)
            result := keccak256(m, 0x1a0)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v13))`.
    function hash(
        bytes32 v0,
        bytes32 v1,
        bytes32 v2,
        bytes32 v3,
        bytes32 v4,
        bytes32 v5,
        bytes32 v6,
        bytes32 v7,
        bytes32 v8,
        bytes32 v9,
        bytes32 v10,
        bytes32 v11,
        bytes32 v12,
        bytes32 v13
    ) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            mstore(add(m, 0xc0), v6)
            mstore(add(m, 0xe0), v7)
            mstore(add(m, 0x100), v8)
            mstore(add(m, 0x120), v9)
            mstore(add(m, 0x140), v10)
            mstore(add(m, 0x160), v11)
            mstore(add(m, 0x180), v12)
            mstore(add(m, 0x1a0), v13)
            result := keccak256(m, 0x1c0)
        }
    }

    /// @dev Returns `keccak256(abi.encode(v0, .., v13))`.
    function hash(
        uint256 v0,
        uint256 v1,
        uint256 v2,
        uint256 v3,
        uint256 v4,
        uint256 v5,
        uint256 v6,
        uint256 v7,
        uint256 v8,
        uint256 v9,
        uint256 v10,
        uint256 v11,
        uint256 v12,
        uint256 v13
    ) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(m, v0)
            mstore(add(m, 0x20), v1)
            mstore(add(m, 0x40), v2)
            mstore(add(m, 0x60), v3)
            mstore(add(m, 0x80), v4)
            mstore(add(m, 0xa0), v5)
            mstore(add(m, 0xc0), v6)
            mstore(add(m, 0xe0), v7)
            mstore(add(m, 0x100), v8)
            mstore(add(m, 0x120), v9)
            mstore(add(m, 0x140), v10)
            mstore(add(m, 0x160), v11)
            mstore(add(m, 0x180), v12)
            mstore(add(m, 0x1a0), v13)
            result := keccak256(m, 0x1c0)
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*             BYTES32 BUFFER HASHING OPERATIONS              */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Returns `keccak256(abi.encode(buffer[0], .., buffer[buffer.length - 1]))`.
    function hash(bytes32[] memory buffer) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := keccak256(add(buffer, 0x20), shl(5, mload(buffer)))
        }
    }

    /// @dev Sets `buffer[i]` to `value`, without a bounds check.
    /// Returns the `buffer` for function chaining.
    function set(bytes32[] memory buffer, uint256 i, bytes32 value)
        internal
        pure
        returns (bytes32[] memory)
    {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(add(buffer, shl(5, add(1, i))), value)
        }
        return buffer;
    }

    /// @dev Sets `buffer[i]` to `value`, without a bounds check.
    /// Returns the `buffer` for function chaining.
    function set(bytes32[] memory buffer, uint256 i, uint256 value)
        internal
        pure
        returns (bytes32[] memory)
    {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(add(buffer, shl(5, add(1, i))), value)
        }
        return buffer;
    }

    /// @dev Returns `new bytes32[](n)`, without zeroing out the memory.
    function malloc(uint256 n) internal pure returns (bytes32[] memory buffer) {
        /// @solidity memory-safe-assembly
        assembly {
            buffer := mload(0x40)
            mstore(buffer, n)
            mstore(0x40, add(shl(5, add(1, n)), buffer))
        }
    }

    /// @dev Frees memory that has been allocated for `buffer`.
    /// No-op if `buffer.length` is zero, or if new memory has been allocated after `buffer`.
    function free(bytes32[] memory buffer) internal pure {
        /// @solidity memory-safe-assembly
        assembly {
            let n := mload(buffer)
            mstore(shl(6, lt(iszero(n), eq(add(shl(5, add(1, n)), buffer), mload(0x40)))), buffer)
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                      EQUALITY CHECKS                       */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Returns `a == abi.decode(b, (bytes32))`.
    function eq(bytes32 a, bytes memory b) internal pure returns (bool result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := and(eq(0x20, mload(b)), eq(a, mload(add(b, 0x20))))
        }
    }

    /// @dev Returns `abi.decode(a, (bytes32)) == b`.
    function eq(bytes memory a, bytes32 b) internal pure returns (bool result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := and(eq(0x20, mload(a)), eq(b, mload(add(a, 0x20))))
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*               BYTE SLICE HASHING OPERATIONS                */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Returns the keccak256 of the slice from `start` to `end` (exclusive).
    /// `start` and `end` are byte offsets.
    function hash(bytes memory b, uint256 start, uint256 end)
        internal
        pure
        returns (bytes32 result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let n := mload(b)
            end := xor(end, mul(xor(end, n), lt(n, end)))
            start := xor(start, mul(xor(start, n), lt(n, start)))
            result := keccak256(add(add(b, 0x20), start), mul(gt(end, start), sub(end, start)))
        }
    }

    /// @dev Returns the keccak256 of the slice from `start` to the end of the bytes.
    function hash(bytes memory b, uint256 start) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let n := mload(b)
            start := xor(start, mul(xor(start, n), lt(n, start)))
            result := keccak256(add(add(b, 0x20), start), mul(gt(n, start), sub(n, start)))
        }
    }

    /// @dev Returns the keccak256 of the bytes.
    function hash(bytes memory b) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := keccak256(add(b, 0x20), mload(b))
        }
    }

    /// @dev Returns the keccak256 of the slice from `start` to `end` (exclusive).
    /// `start` and `end` are byte offsets.
    function hashCalldata(bytes calldata b, uint256 start, uint256 end)
        internal
        pure
        returns (bytes32 result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            end := xor(end, mul(xor(end, b.length), lt(b.length, end)))
            start := xor(start, mul(xor(start, b.length), lt(b.length, start)))
            let n := mul(gt(end, start), sub(end, start))
            calldatacopy(mload(0x40), add(b.offset, start), n)
            result := keccak256(mload(0x40), n)
        }
    }

    /// @dev Returns the keccak256 of the slice from `start` to the end of the bytes.
    function hashCalldata(bytes calldata b, uint256 start) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            start := xor(start, mul(xor(start, b.length), lt(b.length, start)))
            let n := mul(gt(b.length, start), sub(b.length, start))
            calldatacopy(mload(0x40), add(b.offset, start), n)
            result := keccak256(mload(0x40), n)
        }
    }

    /// @dev Returns the keccak256 of the bytes.
    function hashCalldata(bytes calldata b) internal pure returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            calldatacopy(mload(0x40), b.offset, b.length)
            result := keccak256(mload(0x40), b.length)
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                      SHA2-256 HELPERS                      */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Returns `sha256(abi.encode(b))`. Yes, it's more efficient.
    function sha2(bytes32 b) internal view returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, b)
            result := mload(staticcall(gas(), 2, 0x00, 0x20, 0x01, 0x20))
            if iszero(returndatasize()) { invalid() }
        }
    }

    /// @dev Returns the sha256 of the slice from `start` to `end` (exclusive).
    /// `start` and `end` are byte offsets.
    function sha2(bytes memory b, uint256 start, uint256 end)
        internal
        view
        returns (bytes32 result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            let n := mload(b)
            end := xor(end, mul(xor(end, n), lt(n, end)))
            start := xor(start, mul(xor(start, n), lt(n, start)))
            // forgefmt: disable-next-item
            result := mload(staticcall(gas(), 2, add(add(b, 0x20), start),
                mul(gt(end, start), sub(end, start)), 0x01, 0x20))
            if iszero(returndatasize()) { invalid() }
        }
    }

    /// @dev Returns the sha256 of the slice from `start` to the end of the bytes.
    function sha2(bytes memory b, uint256 start) internal view returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let n := mload(b)
            start := xor(start, mul(xor(start, n), lt(n, start)))
            // forgefmt: disable-next-item
            result := mload(staticcall(gas(), 2, add(add(b, 0x20), start),
                mul(gt(n, start), sub(n, start)), 0x01, 0x20))
            if iszero(returndatasize()) { invalid() }
        }
    }

    /// @dev Returns the sha256 of the bytes.
    function sha2(bytes memory b) internal view returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := mload(staticcall(gas(), 2, add(b, 0x20), mload(b), 0x01, 0x20))
            if iszero(returndatasize()) { invalid() }
        }
    }

    /// @dev Returns the sha256 of the slice from `start` to `end` (exclusive).
    /// `start` and `end` are byte offsets.
    function sha2Calldata(bytes calldata b, uint256 start, uint256 end)
        internal
        view
        returns (bytes32 result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            end := xor(end, mul(xor(end, b.length), lt(b.length, end)))
            start := xor(start, mul(xor(start, b.length), lt(b.length, start)))
            let n := mul(gt(end, start), sub(end, start))
            calldatacopy(mload(0x40), add(b.offset, start), n)
            result := mload(staticcall(gas(), 2, mload(0x40), n, 0x01, 0x20))
            if iszero(returndatasize()) { invalid() }
        }
    }

    /// @dev Returns the sha256 of the slice from `start` to the end of the bytes.
    function sha2Calldata(bytes calldata b, uint256 start) internal view returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            start := xor(start, mul(xor(start, b.length), lt(b.length, start)))
            let n := mul(gt(b.length, start), sub(b.length, start))
            calldatacopy(mload(0x40), add(b.offset, start), n)
            result := mload(staticcall(gas(), 2, mload(0x40), n, 0x01, 0x20))
            if iszero(returndatasize()) { invalid() }
        }
    }

    /// @dev Returns the sha256 of the bytes.
    function sha2Calldata(bytes calldata b) internal view returns (bytes32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            calldatacopy(mload(0x40), b.offset, b.length)
            result := mload(staticcall(gas(), 2, mload(0x40), b.length, 0x01, 0x20))
            if iszero(returndatasize()) { invalid() }
        }
    }
}


// ===== FILE: src/nft/IHaNft.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {IERC721Metadata} from "@openzeppelin/contracts/token/ERC721/extensions/IERC721Metadata.sol";
import {IERC721Errors} from "@openzeppelin/contracts/interfaces/draft-IERC6093.sol";

interface IHaNft is IERC721Metadata, IERC721Errors {
    /**
     * @dev Enum to store the result of a reveal
     */
    enum RevealResult {
        CommitmentNotFound,
        AlreadyRevealed,
        MaxSupplyReached,
        RevealBlockNotReached,
        Expired,
        MissingBlockhash,
        InsufficientBalance,
        LimitReached,
        Revealed
    }

    /**
     * @dev Indicates that the contract is already launched
     */
    error AlreadyLaunched();

    /**
     * @dev Indicates that the contract is called by an unauthorized sender
     * @param sender The address that called the contract
     * @param factory The address of the HaFactory contract
     */
    error MsgSenderNotFactory(address sender, address factory);

    /**
     * @dev Indicates that the contract is called by an unauthorized sender
     * @param sender The address that called the contract
     * @param token The address of the HaToken contract
     */
    error MsgSenderNotToken(address sender, address token);

    /**
     * @dev Indicates that the reveal limit is zero
     */
    error LimitIsZero();

    /**
     * @dev Indicates that the commitment is not found
     * @param commitmentIndex The index of the commitment
     */
    error CommitmentNotFound(uint256 commitmentIndex);

    /**
     * @dev Indicates that the commitment is already revealed
     * @param commitmentIndex The index of the commitment
     */
    error CommitmentAlreadyRevealed(uint256 commitmentIndex);

    /**
     * @dev Indicates that the maximum supply is reached
     * @param commitmentIndex The index of the commitment
     * @param mintQuantity The quantity of NFTs minted
     */
    error MaxSupplyReached(uint256 commitmentIndex, uint256 mintQuantity);

    /**
     * @dev Indicates that the reveal block is not reached
     * @param commitmentIndex The index of the commitment
     * @param currentBlock The current block number
     * @param revealBlock The block number at which the NFTs will be revealed
     */
    error RevealBlockNotReached(uint256 commitmentIndex, uint256 currentBlock, uint256 revealBlock);

    /**
     * @dev Indicates that the commitment is expired
     * @param commitmentIndex The index of the commitment
     * @param currentBlock The current block number
     * @param revealBlock The block number at which the NFTs will be revealed
     */
    error CommitmentExpired(uint256 commitmentIndex, uint256 currentBlock, uint256 revealBlock);

    /**
     * @dev Indicates that the blockhash is missing
     * @param commitmentIndex The index of the commitment
     * @param currentBlock The current block number
     * @param revealBlock The block number at which the NFTs will be revealed
     */
    error MissingBlockhash(uint256 commitmentIndex, uint256 currentBlock, uint256 revealBlock);

    /**
     * @dev Indicates that the balance is insufficient
     * @param commitmentIndex The index of the commitment
     * @param mintQuantity The quantity of NFTs minted
     */
    error InsufficientBalance(uint256 commitmentIndex, uint256 mintQuantity);

    /**
     * @dev Emitted when the NFT contract is initialized
     * @param name The name of the NFT
     * @param symbol The symbol of the NFT
     * @param uriStorage The address of the URI storage contract
     * @param maxSupply The maximum supply of the NFT
     * @param factory The address of the HaFactory contract
     * @param token The address of the HaToken contract
     */
    event Initialized(
        string name,
        string symbol,
        address uriStorage,
        uint256 maxSupply,
        address indexed factory,
        address indexed token
    );
    
    /**
     * @dev Emitted when the NFT contract is launched
     * @param pool The address of the Uniswap V4 Pool Manager
     */
    event Launched(address indexed pool);

    /**
     * @dev Emitted when a quantity of NFTs are committed
     * @param commitmentIndex Index of the commitment in the queue
     * @param to The address to receive the revealed NFTs
     * @param quantity The quantity of NFTs committed
     * @param revealBlock The block number at which the NFTs will be revealed
     */
    event Commit(
        uint256 commitmentIndex,
        address indexed to, 
        uint256 quantity,
        uint256 revealBlock
    );

    /**
     * @dev Emitted when a single commitment is revealed
     * @param commitmentIndex The index of the commitment that was revealed
     * @param processedQuantity The quantity of NFTs processed in this commitment
     * @param mintedQuantity The quantity of NFTs minted in this commitment
     * @param result The result of the reveal
     */
    event Reveal(
        uint256 commitmentIndex,
        uint256 processedQuantity,
        uint256 mintedQuantity,
        RevealResult result
    );

    /**
     * @dev Struct to store a commitment to reveal NFTs
     * @param to The address to receive the revealed NFTs
     * @param quantity The quantity of NFTs committed
     * @param processedQuantity The quantity of NFTs already processed in this commitment
     * @param revealBlock The block number at which the NFTs will be revealed
     */
    struct Commitment {
        address to;
        uint256 quantity;
        uint256 processedQuantity;
        uint256 revealBlock;
    }

    /**
     * @notice Returns the Uniswap V4 Pool Manager contract address
     * @return Uniswap V4 Pool Manager contract address
     */
    function pool() external view returns (address);

    /**
     * @notice Returns the associated HaFactory contract address
     * @return HaFactory contract address
     */
    function factory() external view returns (address);

    /**
     * @notice Returns the associated HaToken contract address
     * @return HaToken contract address
     */
    function token() external view returns (address);

    /**
     * @notice Returns the URI storage contract address
     * @return URI storage contract address
     */
    function uriStorage() external view returns (address);

    /**
     * @notice Returns the current number of NFTs minted
     * @return Current NFT supply
     */
    function totalSupply() external view returns (uint256);

    /**
     * @notice Returns the maximum number of NFTs that can be minted
     * @return Maximum NFT supply
     */
    function maxSupply() external view returns (uint256);

    /**
     * @notice Returns the reveal delay
     * @return Reveal delay in blocks
     */
    function revealDelay() external view returns (uint256);

    /**
     * @notice Returns the reveal expiry
     * @return Reveal expiry in blocks
     */
    function revealExpiry() external view returns (uint256);

    /**
     * @notice Initializes the NFT contract
     * @param name_ The name of the NFT
     * @param symbol_ The symbol of the NFT
     * @param uriStorage_ The address of the URI storage contract
     * @param maxSupply_ The maximum supply of the NFT
     * @param factory_ The address of the HaFactory contract
     * @param token_ The address of the HaToken contract
     */
    function initialize(
        string memory name_,
        string memory symbol_,
        address uriStorage_,
        uint256 maxSupply_,
        address factory_,
        address token_
    ) external;

    /**
     * @notice Launches the NFT
     * @dev This function is called by the factory contract
     * @param pool_ The address of the Uniswap V4 Pool Manager
     */
    function launch(address pool_) external;

    /**
     * @notice Commits a quantity of NFTs to be revealed
     * @dev This function is only callable by the associated HaToken contract
     * @param to The address to receive the revealed NFTs
     * @param quantity The quantity of NFTs to commit
     */
    function commit(address to, uint256 quantity) external;

    /**
     * @notice Reveals a quantity of NFTs from the queue
     * @param limit Limit on the number of commitments and reveals that can be processed
     */
    function revealQueue(uint256 limit) external;

    /**
     * @notice Reveals a commit from the queue
     * @param commitmentIndex The index of the commitment to reveal
     * @param limit Limit on the number of NFTs to reveal
     */
    function reveal(uint256 commitmentIndex, uint256 limit) external;

    /**
     * @notice Moves a quantity of NFTs from one address to another
     * @dev This function is only callable by the associated HaToken contract
     * @param from The address to move the NFTs from
     * @param to The address to move the NFTs to
     * @param quantity The quantity of NFTs to move
     */
    function move(address from, address to, uint256 quantity) external;

    /**
     * @notice Burns a quantity of NFTs from an address
     * @dev This function is only callable by the associated HaToken contract
     * @param from The address to burn the NFTs from
     * @param quantity The quantity of NFTs to burn
     */
    function burn(address from, uint256 quantity) external;
}


// ===== FILE: src/nft/IUriStorage.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

interface IUriStorage {
    /**
     * @notice Returns the URI for a token
     * @param tokenId The ID of the token
     * @return The URI for the token
     */
    function tokenURI(uint256 tokenId) external view returns (string memory);

    /**
     * @notice Draws a token ID
     * @param random The random number
     * @param totalSupply The total supply of the NFT
     * @param maxSupply The maximum supply of the NFT
     * @return tokenId The token ID
     */
    function drawTokenId(
        bytes32 random,
        uint256 totalSupply, 
        uint256 maxSupply
    ) external returns (uint256 tokenId);

    /**
     * @notice Returns a token ID
     * @param tokenId The ID of the token
     * @param totalSupply The total supply of the NFT
     * @param maxSupply The maximum supply of the NFT
     */
    function returnTokenId(
        uint256 tokenId, 
        uint256 totalSupply, 
        uint256 maxSupply
    ) external;
}

// ===== FILE: lib/openzeppelin-contracts/contracts/token/ERC721/extensions/IERC721Metadata.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (token/ERC721/extensions/IERC721Metadata.sol)

pragma solidity >=0.6.2;

import {IERC721} from "../IERC721.sol";

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


// ===== FILE: lib/openzeppelin-contracts/contracts/interfaces/draft-IERC6093.sol =====
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


// ===== FILE: lib/openzeppelin-contracts/contracts/token/ERC721/IERC721.sol =====
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


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/introspection/IERC165.sol =====
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
