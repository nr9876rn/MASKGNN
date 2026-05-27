// ===== FILE: npm/_openzeppelin/contracts-upgradeable_5.4.0/proxy/utils/Initializable.sol =====
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


// ===== FILE: npm/_openzeppelin/contracts-upgradeable_5.4.0/utils/ContextUpgradeable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.1) (utils/Context.sol)

pragma solidity ^0.8.20;
import {Initializable} from "../proxy/utils/Initializable.sol";

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


// ===== FILE: npm/solidity-bytes-utils_0.8.4/contracts/BytesLib.sol =====
// SPDX-License-Identifier: Unlicense
/*
 * @title Solidity Bytes Arrays Utils
 * @author Gonçalo Sá <goncalo.sa@consensys.net>
 *
 * @dev Bytes tightly packed arrays utility library for ethereum contracts written in Solidity.
 *      The library lets you concatenate, slice and type cast bytes arrays both in memory and storage.
 */
pragma solidity >=0.8.0 <0.9.0;


library BytesLib {
    function concat(
        bytes memory _preBytes,
        bytes memory _postBytes
    )
        internal
        pure
        returns (bytes memory)
    {
        bytes memory tempBytes;

        assembly {
            // Get a location of some free memory and store it in tempBytes as
            // Solidity does for memory variables.
            tempBytes := mload(0x40)

            // Store the length of the first bytes array at the beginning of
            // the memory for tempBytes.
            let length := mload(_preBytes)
            mstore(tempBytes, length)

            // Maintain a memory counter for the current write location in the
            // temp bytes array by adding the 32 bytes for the array length to
            // the starting location.
            let mc := add(tempBytes, 0x20)
            // Stop copying when the memory counter reaches the length of the
            // first bytes array.
            let end := add(mc, length)

            for {
                // Initialize a copy counter to the start of the _preBytes data,
                // 32 bytes into its memory.
                let cc := add(_preBytes, 0x20)
            } lt(mc, end) {
                // Increase both counters by 32 bytes each iteration.
                mc := add(mc, 0x20)
                cc := add(cc, 0x20)
            } {
                // Write the _preBytes data into the tempBytes memory 32 bytes
                // at a time.
                mstore(mc, mload(cc))
            }

            // Add the length of _postBytes to the current length of tempBytes
            // and store it as the new length in the first 32 bytes of the
            // tempBytes memory.
            length := mload(_postBytes)
            mstore(tempBytes, add(length, mload(tempBytes)))

            // Move the memory counter back from a multiple of 0x20 to the
            // actual end of the _preBytes data.
            mc := end
            // Stop copying when the memory counter reaches the new combined
            // length of the arrays.
            end := add(mc, length)

            for {
                let cc := add(_postBytes, 0x20)
            } lt(mc, end) {
                mc := add(mc, 0x20)
                cc := add(cc, 0x20)
            } {
                mstore(mc, mload(cc))
            }

            // Update the free-memory pointer by padding our last write location
            // to 32 bytes: add 31 bytes to the end of tempBytes to move to the
            // next 32 byte block, then round down to the nearest multiple of
            // 32. If the sum of the length of the two arrays is zero then add
            // one before rounding down to leave a blank 32 bytes (the length block with 0).
            mstore(0x40, and(
              add(add(end, iszero(add(length, mload(_preBytes)))), 31),
              not(31) // Round down to the nearest 32 bytes.
            ))
        }

        return tempBytes;
    }

    function concatStorage(bytes storage _preBytes, bytes memory _postBytes) internal {
        assembly {
            // Read the first 32 bytes of _preBytes storage, which is the length
            // of the array. (We don't need to use the offset into the slot
            // because arrays use the entire slot.)
            let fslot := sload(_preBytes.slot)
            // Arrays of 31 bytes or less have an even value in their slot,
            // while longer arrays have an odd value. The actual length is
            // the slot divided by two for odd values, and the lowest order
            // byte divided by two for even values.
            // If the slot is even, bitwise and the slot with 255 and divide by
            // two to get the length. If the slot is odd, bitwise and the slot
            // with -1 and divide by two.
            let slength := div(and(fslot, sub(mul(0x100, iszero(and(fslot, 1))), 1)), 2)
            let mlength := mload(_postBytes)
            let newlength := add(slength, mlength)
            // slength can contain both the length and contents of the array
            // if length < 32 bytes so let's prepare for that
            // v. http://solidity.readthedocs.io/en/latest/miscellaneous.html#layout-of-state-variables-in-storage
            switch add(lt(slength, 32), lt(newlength, 32))
            case 2 {
                // Since the new array still fits in the slot, we just need to
                // update the contents of the slot.
                // uint256(bytes_storage) = uint256(bytes_storage) + uint256(bytes_memory) + new_length
                sstore(
                    _preBytes.slot,
                    // all the modifications to the slot are inside this
                    // next block
                    add(
                        // we can just add to the slot contents because the
                        // bytes we want to change are the LSBs
                        fslot,
                        add(
                            mul(
                                div(
                                    // load the bytes from memory
                                    mload(add(_postBytes, 0x20)),
                                    // zero all bytes to the right
                                    exp(0x100, sub(32, mlength))
                                ),
                                // and now shift left the number of bytes to
                                // leave space for the length in the slot
                                exp(0x100, sub(32, newlength))
                            ),
                            // increase length by the double of the memory
                            // bytes length
                            mul(mlength, 2)
                        )
                    )
                )
            }
            case 1 {
                // The stored value fits in the slot, but the combined value
                // will exceed it.
                // get the keccak hash to get the contents of the array
                mstore(0x0, _preBytes.slot)
                let sc := add(keccak256(0x0, 0x20), div(slength, 32))

                // save new length
                sstore(_preBytes.slot, add(mul(newlength, 2), 1))

                // The contents of the _postBytes array start 32 bytes into
                // the structure. Our first read should obtain the `submod`
                // bytes that can fit into the unused space in the last word
                // of the stored array. To get this, we read 32 bytes starting
                // from `submod`, so the data we read overlaps with the array
                // contents by `submod` bytes. Masking the lowest-order
                // `submod` bytes allows us to add that value directly to the
                // stored value.

                let submod := sub(32, slength)
                let mc := add(_postBytes, submod)
                let end := add(_postBytes, mlength)
                let mask := sub(exp(0x100, submod), 1)

                sstore(
                    sc,
                    add(
                        and(
                            fslot,
                            0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff00
                        ),
                        and(mload(mc), mask)
                    )
                )

                for {
                    mc := add(mc, 0x20)
                    sc := add(sc, 1)
                } lt(mc, end) {
                    sc := add(sc, 1)
                    mc := add(mc, 0x20)
                } {
                    sstore(sc, mload(mc))
                }

                mask := exp(0x100, sub(mc, end))

                sstore(sc, mul(div(mload(mc), mask), mask))
            }
            default {
                // get the keccak hash to get the contents of the array
                mstore(0x0, _preBytes.slot)
                // Start copying to the last used word of the stored array.
                let sc := add(keccak256(0x0, 0x20), div(slength, 32))

                // save new length
                sstore(_preBytes.slot, add(mul(newlength, 2), 1))

                // Copy over the first `submod` bytes of the new data as in
                // case 1 above.
                let slengthmod := mod(slength, 32)
                let mlengthmod := mod(mlength, 32)
                let submod := sub(32, slengthmod)
                let mc := add(_postBytes, submod)
                let end := add(_postBytes, mlength)
                let mask := sub(exp(0x100, submod), 1)

                sstore(sc, add(sload(sc), and(mload(mc), mask)))

                for {
                    sc := add(sc, 1)
                    mc := add(mc, 0x20)
                } lt(mc, end) {
                    sc := add(sc, 1)
                    mc := add(mc, 0x20)
                } {
                    sstore(sc, mload(mc))
                }

                mask := exp(0x100, sub(mc, end))

                sstore(sc, mul(div(mload(mc), mask), mask))
            }
        }
    }

    function slice(
        bytes memory _bytes,
        uint256 _start,
        uint256 _length
    )
        internal
        pure
        returns (bytes memory)
    {
        // We're using the unchecked block below because otherwise execution ends 
        // with the native overflow error code.
        unchecked {
            require(_length + 31 >= _length, "slice_overflow");
        }
        require(_bytes.length >= _start + _length, "slice_outOfBounds");

        bytes memory tempBytes;

        assembly {
            switch iszero(_length)
            case 0 {
                // Get a location of some free memory and store it in tempBytes as
                // Solidity does for memory variables.
                tempBytes := mload(0x40)

                // The first word of the slice result is potentially a partial
                // word read from the original array. To read it, we calculate
                // the length of that partial word and start copying that many
                // bytes into the array. The first word we copy will start with
                // data we don't care about, but the last `lengthmod` bytes will
                // land at the beginning of the contents of the new array. When
                // we're done copying, we overwrite the full first word with
                // the actual length of the slice.
                let lengthmod := and(_length, 31)

                // The multiplication in the next line is necessary
                // because when slicing multiples of 32 bytes (lengthmod == 0)
                // the following copy loop was copying the origin's length
                // and then ending prematurely not copying everything it should.
                let mc := add(add(tempBytes, lengthmod), mul(0x20, iszero(lengthmod)))
                let end := add(mc, _length)

                for {
                    // The multiplication in the next line has the same exact purpose
                    // as the one above.
                    let cc := add(add(add(_bytes, lengthmod), mul(0x20, iszero(lengthmod))), _start)
                } lt(mc, end) {
                    mc := add(mc, 0x20)
                    cc := add(cc, 0x20)
                } {
                    mstore(mc, mload(cc))
                }

                mstore(tempBytes, _length)

                //update free-memory pointer
                //allocating the array padded to 32 bytes like the compiler does now
                mstore(0x40, and(add(mc, 31), not(31)))
            }
            //if we want a zero-length slice let's just return a zero-length array
            default {
                tempBytes := mload(0x40)
                //zero out the 32 bytes slice we are about to return
                //we need to do it because Solidity does not garbage collect
                mstore(tempBytes, 0)

                mstore(0x40, add(tempBytes, 0x20))
            }
        }

        return tempBytes;
    }

    function toAddress(bytes memory _bytes, uint256 _start) internal pure returns (address) {
        require(_bytes.length >= _start + 20, "toAddress_outOfBounds");
        address tempAddress;

        assembly {
            tempAddress := div(mload(add(add(_bytes, 0x20), _start)), 0x1000000000000000000000000)
        }

        return tempAddress;
    }

    function toUint8(bytes memory _bytes, uint256 _start) internal pure returns (uint8) {
        require(_bytes.length >= _start + 1 , "toUint8_outOfBounds");
        uint8 tempUint;

        assembly {
            tempUint := mload(add(add(_bytes, 0x1), _start))
        }

        return tempUint;
    }

    function toUint16(bytes memory _bytes, uint256 _start) internal pure returns (uint16) {
        require(_bytes.length >= _start + 2, "toUint16_outOfBounds");
        uint16 tempUint;

        assembly {
            tempUint := mload(add(add(_bytes, 0x2), _start))
        }

        return tempUint;
    }

    function toUint32(bytes memory _bytes, uint256 _start) internal pure returns (uint32) {
        require(_bytes.length >= _start + 4, "toUint32_outOfBounds");
        uint32 tempUint;

        assembly {
            tempUint := mload(add(add(_bytes, 0x4), _start))
        }

        return tempUint;
    }

    function toUint64(bytes memory _bytes, uint256 _start) internal pure returns (uint64) {
        require(_bytes.length >= _start + 8, "toUint64_outOfBounds");
        uint64 tempUint;

        assembly {
            tempUint := mload(add(add(_bytes, 0x8), _start))
        }

        return tempUint;
    }

    function toUint96(bytes memory _bytes, uint256 _start) internal pure returns (uint96) {
        require(_bytes.length >= _start + 12, "toUint96_outOfBounds");
        uint96 tempUint;

        assembly {
            tempUint := mload(add(add(_bytes, 0xc), _start))
        }

        return tempUint;
    }

    function toUint128(bytes memory _bytes, uint256 _start) internal pure returns (uint128) {
        require(_bytes.length >= _start + 16, "toUint128_outOfBounds");
        uint128 tempUint;

        assembly {
            tempUint := mload(add(add(_bytes, 0x10), _start))
        }

        return tempUint;
    }

    function toUint256(bytes memory _bytes, uint256 _start) internal pure returns (uint256) {
        require(_bytes.length >= _start + 32, "toUint256_outOfBounds");
        uint256 tempUint;

        assembly {
            tempUint := mload(add(add(_bytes, 0x20), _start))
        }

        return tempUint;
    }

    function toBytes32(bytes memory _bytes, uint256 _start) internal pure returns (bytes32) {
        require(_bytes.length >= _start + 32, "toBytes32_outOfBounds");
        bytes32 tempBytes32;

        assembly {
            tempBytes32 := mload(add(add(_bytes, 0x20), _start))
        }

        return tempBytes32;
    }

    function equal(bytes memory _preBytes, bytes memory _postBytes) internal pure returns (bool) {
        bool success = true;

        assembly {
            let length := mload(_preBytes)

            // if lengths don't match the arrays are not equal
            switch eq(length, mload(_postBytes))
            case 1 {
                // cb is a circuit breaker in the for loop since there's
                //  no said feature for inline assembly loops
                // cb = 1 - don't breaker
                // cb = 0 - break
                let cb := 1

                let mc := add(_preBytes, 0x20)
                let end := add(mc, length)

                for {
                    let cc := add(_postBytes, 0x20)
                // the next line is the loop condition:
                // while(uint256(mc < end) + cb == 2)
                } eq(add(lt(mc, end), cb), 2) {
                    mc := add(mc, 0x20)
                    cc := add(cc, 0x20)
                } {
                    // if any of these checks fails then arrays are not equal
                    if iszero(eq(mload(mc), mload(cc))) {
                        // unsuccess:
                        success := 0
                        cb := 0
                    }
                }
            }
            default {
                // unsuccess:
                success := 0
            }
        }

        return success;
    }

    function equalStorage(
        bytes storage _preBytes,
        bytes memory _postBytes
    )
        internal
        view
        returns (bool)
    {
        bool success = true;

        assembly {
            // we know _preBytes_offset is 0
            let fslot := sload(_preBytes.slot)
            // Decode the length of the stored array like in concatStorage().
            let slength := div(and(fslot, sub(mul(0x100, iszero(and(fslot, 1))), 1)), 2)
            let mlength := mload(_postBytes)

            // if lengths don't match the arrays are not equal
            switch eq(slength, mlength)
            case 1 {
                // slength can contain both the length and contents of the array
                // if length < 32 bytes so let's prepare for that
                // v. http://solidity.readthedocs.io/en/latest/miscellaneous.html#layout-of-state-variables-in-storage
                if iszero(iszero(slength)) {
                    switch lt(slength, 32)
                    case 1 {
                        // blank the last byte which is the length
                        fslot := mul(div(fslot, 0x100), 0x100)

                        if iszero(eq(fslot, mload(add(_postBytes, 0x20)))) {
                            // unsuccess:
                            success := 0
                        }
                    }
                    default {
                        // cb is a circuit breaker in the for loop since there's
                        //  no said feature for inline assembly loops
                        // cb = 1 - don't breaker
                        // cb = 0 - break
                        let cb := 1

                        // get the keccak hash to get the contents of the array
                        mstore(0x0, _preBytes.slot)
                        let sc := keccak256(0x0, 0x20)

                        let mc := add(_postBytes, 0x20)
                        let end := add(mc, mlength)

                        // the next line is the loop condition:
                        // while(uint256(mc < end) + cb == 2)
                        for {} eq(add(lt(mc, end), cb), 2) {
                            sc := add(sc, 1)
                            mc := add(mc, 0x20)
                        } {
                            if iszero(eq(sload(sc), mload(mc))) {
                                // unsuccess:
                                success := 0
                                cb := 0
                            }
                        }
                    }
                }
            }
            default {
                // unsuccess:
                success := 0
            }
        }

        return success;
    }
}


// ===== FILE: project/contracts/interfaces/IAuthValidator.sol =====
// SPDX-License-Identifier: GPL-3.0
pragma solidity 0.8.27;

/**
 * @dev IAuthValidator. Interface for verification of auth data.
 */
interface IAuthValidator {
    /**
     * @dev AuthResponseField. Information about response fields from verification. Used in verify function.
     * @param name Name of the response field
     * @param value Value of the response field
     */
    struct AuthResponseField {
        string name;
        uint256 value;
    }

    /**
     * @dev Get version of the contract
     */
    function version() external view returns (string memory);

    /**
     * @dev Verify the proof with the supported method informed in the auth query data
     * packed as bytes and that the proof was generated by the sender.
     * @param sender Sender of the proof.
     * @param proof Proof packed as bytes to verify.
     * @param params Request query data of the credential to verify.
     * @return userID User Id for the auth proof verified and response fields.
     * @return authResponseFields Additional response fields.
     */
    function verify(
        address sender,
        bytes calldata proof,
        bytes calldata params
    ) external returns (uint256 userID, AuthResponseField[] memory authResponseFields);
}


// ===== FILE: project/contracts/interfaces/IRequestValidator.sol =====
// SPDX-License-Identifier: GPL-3.0
pragma solidity 0.8.27;

/**
 * @dev IRequestValidator. Interface for verification of request query data.
 */
interface IRequestValidator {
    error RequestParamNameNotFound();
    error InputNameNotFound();
    /**
     * @dev ResponseField. Information about response fields from verification. Used in verify function.
     * @param name Name of the response field
     * @param value Value of the response field
     * @param rawValue Raw value of the response field
     */
    struct ResponseField {
        string name;
        uint256 value;
        bytes rawValue;
    }

    /**
     * @dev RequestParam. Information about request param from request query data.
     * @param name Name of the request query param
     * @param value Value of the request query param
     */
    struct RequestParam {
        string name;
        uint256 value;
    }

    /**
     * @dev Get version of the contract
     */
    function version() external view returns (string memory);

    /**
     * @dev Verify the proof with the supported method informed in the request query data
     * packed as bytes and that the proof was generated by the sender.
     * @param sender Sender of the proof.
     * @param proof Proof packed as bytes to verify.
     * @param requestParams Request query data of the credential to verify.
     * @param responseMetadata Metadata from the response.
     * @return Array of response fields as result.
     */
    function verify(
        address sender,
        bytes calldata proof,
        bytes calldata requestParams,
        bytes calldata responseMetadata
    ) external returns (ResponseField[] memory);

    /**
     * @dev Get the request param from params of the request query data.
     * @param params Request query data of the credential to verify.
     * @param paramName Request query param name to retrieve of the credential to verify.
     * @return RequestParam for the param name of the request query data.
     */
    function getRequestParam(
        bytes calldata params,
        string memory paramName
    ) external view returns (RequestParam memory);

    /**
     * @dev Get the index of the public input of the circuit by name
     * @param name Name of the public input
     * @return Index of the public input
     */
    function inputIndexOf(string memory name) external view returns (uint256);
}


// ===== FILE: project/contracts/interfaces/IState.sol =====
// SPDX-License-Identifier: GPL-3.0
pragma solidity 0.8.27;

// TODO changing this value don't forget to change GistProof.siblings array size
// figure out how to reuse the constant in the array size
// without compiler error
uint256 constant MAX_SMT_DEPTH = 64;

interface IState {
    /**
     * @dev Struct for public interfaces to represent a state information.
     * @param id An identity.
     * @param state A state.
     * @param replacedByState A state, which replaced this state for the identity.
     * @param createdAtTimestamp A time when the state was created.
     * @param replacedAtTimestamp A time when the state was replaced by the next identity state.
     * @param createdAtBlock A block number when the state was created.
     * @param replacedAtBlock A block number when the state was replaced by the next identity state.
     */
    struct StateInfo {
        uint256 id;
        uint256 state;
        uint256 replacedByState;
        uint256 createdAtTimestamp;
        uint256 replacedAtTimestamp;
        uint256 createdAtBlock;
        uint256 replacedAtBlock;
    }

    /**
     * @dev Struct for public interfaces to represent GIST root information.
     * @param root This GIST root.
     * @param replacedByRoot A root, which replaced this root.
     * @param createdAtTimestamp A time, when the root was saved to blockchain.
     * @param replacedAtTimestamp A time, when the root was replaced by the next root in blockchain.
     * @param createdAtBlock A number of block, when the root was saved to blockchain.
     * @param replacedAtBlock A number of block, when the root was replaced by the next root in blockchain.
     */
    struct GistRootInfo {
        uint256 root;
        uint256 replacedByRoot;
        uint256 createdAtTimestamp;
        uint256 replacedAtTimestamp;
        uint256 createdAtBlock;
        uint256 replacedAtBlock;
    }

    /**
     * @dev Struct for public interfaces to represent GIST proof information.
     * @param root This GIST root.
     * @param existence A flag, which shows if the leaf index exists in the GIST.
     * @param siblings An array of GIST sibling node hashes.
     * @param index An index of the leaf in the GIST.
     * @param value A value of the leaf in the GIST.
     * @param auxExistence A flag, which shows if the auxiliary leaf exists in the GIST.
     * @param auxIndex An index of the auxiliary leaf in the GIST.
     * @param auxValue A value of the auxiliary leaf in the GIST.
     */
    struct GistProof {
        uint256 root;
        bool existence;
        uint256[64] siblings; // TODO figure out the way how to use the MAX_SMT_DEPTH constant
        uint256 index;
        uint256 value;
        bool auxExistence;
        uint256 auxIndex;
        uint256 auxValue;
    }
    /**
     * @dev Struct for signed identity states.
     * @param idStateMsg Message of the identity state.
     * @param signature Signature of the message.
     */
    struct IdentityStateUpdate {
        IdentityStateMessage idStateMsg;
        bytes signature;
    }
    /**
     * @dev Struct for signed global states.
     * @param globalStateMsg Message of the global state.
     * @param signature Signature of the message.
     */
    struct GlobalStateUpdate {
        GlobalStateMessage globalStateMsg;
        bytes signature;
    }
    /**
     * @dev Struct for identity state message.
     * @param timestamp Timestamp when the message was signed.
     * @param id Id of the identity.
     * @param state State of the identity.
     * @param replacedAtTimestamp Timestamp when the state was replaced by next identity state.
     */
    struct IdentityStateMessage {
        uint256 timestamp;
        uint256 id;
        uint256 state;
        uint256 replacedAtTimestamp;
    }

    /**
     * @dev Struct for global state message.
     * @param timestamp Timestamp when the message was signed.
     * @param idType Id type of the chain.
     * @param root Root of the global state.
     * @param replacedAtTimestamp Timestamp when the global state was replaced by next global state.
     */
    struct GlobalStateMessage {
        uint256 timestamp;
        bytes2 idType;
        uint256 root;
        uint256 replacedAtTimestamp;
    }
    /**
     * @dev Struct for cross chain proof.
     * @param proofType Proof type for the proof provided ("stateProof", "globalStateProof").
     * @param proof Cross chain proof.
     */
    struct CrossChainProof {
        string proofType;
        bytes proof;
    }

    /**
     * @dev Struct for global state process result.
     * @param idType Id type of the chain.
     * @param root Root of the global state.
     * @param replacedAtTimestamp Timestamp when the global state was replaced by next global state.
     */
    struct GlobalStateProcessResult {
        bytes2 idType;
        uint256 root;
        uint256 replacedAtTimestamp;
    }
    /**
     * @dev Struct for identity state process result.
     * @param id Id of the identity.
     * @param state State of the identity.
     * @param replacedAtTimestamp Timestamp when the identity state was replaced by next identity state.
     */
    struct IdentityStateProcessResult {
        uint256 id;
        uint256 state;
        uint256 replacedAtTimestamp;
    }

    /**
     * @dev Retrieve last state information of specific id.
     * @param id An identity.
     * @return The state info.
     */
    function getStateInfoById(uint256 id) external view returns (StateInfo memory);

    /**
     * @dev Retrieve state information by id and state.
     * @param id An identity.
     * @param state A state.
     * @return The state info.
     */
    function getStateInfoByIdAndState(
        uint256 id,
        uint256 state
    ) external view returns (StateInfo memory);

    /**
     * @dev Retrieve the specific GIST root information.
     * @param root GIST root.
     * @return The GIST root info.
     */
    function getGISTRootInfo(uint256 root) external view returns (GistRootInfo memory);

    /**
     * @dev Check if the id type supported.
     * @param idType id type.
     * @return True if the id type supported.
     */
    function isIdTypeSupported(bytes2 idType) external view returns (bool);

    /**
     * @dev Get id if the id type supported for the id, otherwise revert.
     * @param id An identity.
     * @return The id type.
     */
    function getIdTypeIfSupported(uint256 id) external view returns (bytes2);

    /**
     * @dev Get defaultIdType
     * @return defaultIdType
     */
    function getDefaultIdType() external view returns (bytes2);

    /**
     * @dev Performs state transition
     * @param id Identifier of the identity
     * @param oldState Previous state of the identity
     * @param newState New state of the identity
     * @param isOldStateGenesis Flag if previous identity state is genesis
     * @param a Proof.A
     * @param b Proof.B
     * @param c Proof.C
     */
    function transitState(
        uint256 id,
        uint256 oldState,
        uint256 newState,
        bool isOldStateGenesis,
        uint256[2] memory a,
        uint256[2][2] memory b,
        uint256[2] memory c
    ) external;

    /**
     * @dev Performs state transition
     * @param id Identity
     * @param oldState Previous identity state
     * @param newState New identity state
     * @param isOldStateGenesis Is the previous state genesis?
     * @param methodId State transition method id
     * @param methodParams State transition method-specific params
     */
    function transitStateGeneric(
        uint256 id,
        uint256 oldState,
        uint256 newState,
        bool isOldStateGenesis,
        uint256 methodId,
        bytes calldata methodParams
    ) external;

    /**
     * @dev Check if identity exists.
     * @param id Identity
     * @return True if the identity exists
     */
    function idExists(uint256 id) external view returns (bool);

    /**
     * @dev Check if state exists.
     * @param id Identity
     * @param state State
     * @return True if the state exists
     */
    function stateExists(uint256 id, uint256 state) external view returns (bool);

    /**
     * @dev Get timestamp when the identity state was replaced.
     * @param id Identity
     * @param state State of the identity
     * @return replacedAtTimestamp Timestamp when the identity state was replaced by new identity state
     */
    function getStateReplacedAt(
        uint256 id,
        uint256 state
    ) external view returns (uint256 replacedAtTimestamp);

    /**
     * @dev Get timestamp when the global state was replaced.
     * @param idType Id type of the chain
     * @param root Root of the global state
     * @return replacedAtTimestamp Timestamp when the global state was replaced by new global state
     */
    function getGistRootReplacedAt(
        bytes2 idType,
        uint256 root
    ) external view returns (uint256 replacedAtTimestamp);

    /**
     * @dev Process the cross chain proofs with the identities and global states.
     * @param proofs Proofs with the identities and global states
     */
    function processCrossChainProofs(bytes calldata proofs) external;
}


// ===== FILE: project/contracts/interfaces/IVerifier.sol =====
// SPDX-License-Identifier: GPL-3.0

pragma solidity 0.8.27;

import {IAuthValidator} from "./IAuthValidator.sol";
import {IRequestValidator} from "./IRequestValidator.sol";

/**
 * @dev IVerifier. Interface for creating requests and verifying request responses through validators circuits.
 */
interface IVerifier {
    /**
     * @dev Request. Structure for setting request.
     * @param requestId Request id.
     * @param metadata Metadata of the request.
     * @param validator Validator to verify the response.
     * @param params Parameters data of the request.
     * @param creator Creator of the request.
     */
    struct Request {
        uint256 requestId;
        string metadata;
        IRequestValidator validator;
        bytes params;
        address creator;
    }

    /**
     * @dev Request. Structure for request for storage.
     * @param metadata Metadata of the request.
     * @param validator Validator circuit.
     * @param params Params of the request. Proof parameters could be ZK groth16, plonk, ESDSA, EIP712, etc.
     * @param creator Creator of the request.
     */
    struct RequestData {
        string metadata;
        IRequestValidator validator;
        bytes params;
        address creator;
    }

    /**
     * @dev RequestInfo. Structure for getting request info.
     * @param requestId Request id.
     * @param metadata Metadata of the request.
     * @param validator Validator to verify the response.
     * @param params Parameters data of the request.
     * @param creator Creator of the request.
     */
    struct RequestInfo {
        uint256 requestId;
        string metadata;
        IRequestValidator validator;
        bytes params;
        address creator;
    }

    /**
     * @dev Response. Structure for response.
     * @param requestId Request id of the request.
     * @param proof proof to verify.
     * @param metadata Metadata of the request.
     */
    struct Response {
        uint256 requestId;
        bytes proof;
        bytes metadata;
    }

    /**
     * @dev AuthResponse. Structure for auth response.
     * @param authMethod Auth type of the proof response.
     * @param proof proof to verify.
     */
    struct AuthResponse {
        string authMethod;
        bytes proof;
    }

    /**
     * @dev RequestProofStatus. Structure for request proof status.
     * @param requestId Request id of the proof.
     * @param isVerified True if the proof is verified.
     * @param validatorVersion Version of the validator.
     * @param timestamp Timestamp of the proof.
     */
    struct RequestProofStatus {
        uint256 requestId;
        bool isVerified;
        string validatorVersion;
        uint256 timestamp;
    }

    /**
     * @dev AuthMethod. Structure for auth type for auth proofs.
     * @param authMethod Auth type of the auth proof.
     * @param validator Validator to verify the auth.
     * @param params Parameters data of the auth.
     */
    struct AuthMethod {
        string authMethod;
        IAuthValidator validator;
        bytes params;
    }

    /**
     * @dev MultiRequest. Structure for multiRequest.
     * @param multiRequestId MultiRequest id.
     * @param requestIds Request ids for this multi multiRequest (without groupId. Single requests).
     * @param groupIds Group ids for this multi multiRequest (all the requests included in the group. Grouped requests).
     * @param metadata Metadata for the multiRequest. Empty in first version.
     */
    struct MultiRequest {
        uint256 multiRequestId;
        uint256[] requestIds;
        uint256[] groupIds;
        bytes metadata;
    }

    /**
     * @dev Submits an array of responses and updates proofs status
     * @param authResponse Auth response including auth type and proof
     * @param responses The list of responses including request ID, proof and metadata for requests
     * @param crossChainProofs The list of cross chain proofs from universal resolver (oracle). This
     * includes identities and global states.
     */
    function submitResponse(
        AuthResponse memory authResponse,
        Response[] memory responses,
        bytes memory crossChainProofs
    ) external;

    /**
     * @dev Sets different requests
     * @param requests List of requests
     */
    function setRequests(Request[] calldata requests) external;

    /**
     * @dev Gets a specific request by ID
     * @param requestId The ID of the request
     * @return request The request info
     */
    function getRequest(uint256 requestId) external view returns (RequestInfo memory request);

    /**
     * @dev Get the requests count.
     * @return Requests count.
     */
    function getRequestsCount() external view returns (uint256);

    /**
     * @dev Get the group of requests count.
     * @return Group of requests count.
     */
    function getGroupsCount() external view returns (uint256);

    /**
     * @dev Get the group of requests.
     * @return Group of requests.
     */
    function getGroupedRequests(uint256 groupID) external view returns (RequestInfo[] memory);

    /**
     * @dev Checks if a request ID exists
     * @param requestId The ID of the request
     * @return Whether the request ID exists
     */
    function requestIdExists(uint256 requestId) external view returns (bool);

    /**
     * @dev Checks if a group ID exists
     * @param groupId The ID of the group
     * @return Whether the group ID exists
     */
    function groupIdExists(uint256 groupId) external view returns (bool);

    /**
     * @dev Checks if a multiRequest ID exists
     * @param multiRequestId The ID of the multiRequest
     * @return Whether the multiRequest ID exists
     */
    function multiRequestIdExists(uint256 multiRequestId) external view returns (bool);

    /**
     * @dev Gets the status of the multiRequest verification
     * @param multiRequestId The ID of the MultiRequest
     * @param userAddress The address of the user
     * @return status The status of the MultiRequest. "True" if all requests are verified, "false" otherwise
     */
    function getMultiRequestProofsStatus(
        uint256 multiRequestId,
        address userAddress
    ) external view returns (RequestProofStatus[] memory);

    /**
     * @dev Checks if the proofs from a Multirequest submitted for a given sender and request ID are verified
     * @param multiRequestId The ID of the MultiRequest
     * @param userAddress The address of the user
     * @return Wether the multiRequest is verified.
     */
    function areMultiRequestProofsVerified(
        uint256 multiRequestId,
        address userAddress
    ) external view returns (bool);

    /**
     * @dev Gets proof storage response field value
     * @param requestId Id of the request
     * @param sender Address of the user
     * @param responseFieldName Name of the proof storage response field to get
     * @return The value of the proof storage response field for the user address
     */
    function getResponseFieldValue(
        uint256 requestId,
        address sender,
        string memory responseFieldName
    ) external view returns (uint256);

    /**
     * @dev Gets proof storage response field value by user ID
     * @param requestId Id of the request
     * @param userId ID of the user
     * @param responseFieldName Name of the proof storage response field to get
     * @return The value of the proof storage response field for the user ID
     */
    function getResponseFieldValueByUserId(
        uint256 requestId,
        uint256 userId,
        string memory responseFieldName
    ) external view returns (uint256);

    /**
     * @dev Gets proof storage response fields
     * @param requestId Id of the request
     * @param sender Address of the user
     */
    function getResponseFields(
        uint256 requestId,
        address sender
    ) external view returns (IRequestValidator.ResponseField[] memory);

    /**
     * @dev Checks if a proof from a request submitted for a given sender and request ID is verified
     * @param sender Sender of the proof.
     * @param requestId Request id of the Request to verify.
     * @return True if proof is verified for the sender and request id.
     */
    function isRequestProofVerified(address sender, uint256 requestId) external view returns (bool);

    /**
     * @dev Checks if a proof from a request submitted for a given user ID and request ID is verified
     * @param userId ID of the user.
     * @param requestId Request id of the Request to verify.
     * @return True if proof is verified for the user ID and request id.
     */
    function isRequestProofVerifiedByUserId(
        uint256 userId,
        uint256 requestId
    ) external view returns (bool);

    /**
     * @dev Sets an auth method
     * @param authMethod The auth method to add
     */
    function setAuthMethod(AuthMethod calldata authMethod) external;

    /**
     * @dev Sets a multiRequest
     * @param multiRequest The multiRequest data
     */
    function setMultiRequest(MultiRequest calldata multiRequest) external;

    /**
     * @dev Gets a specific multiRequest by ID
     * @param multiRequestId The ID of the multiRequest
     * @return multiRequest The multiRequest data
     */
    function getMultiRequest(
        uint256 multiRequestId
    ) external view returns (MultiRequest memory multiRequest);

    /**
     * @dev Get the proof status for the sender and request with requestId.
     * @param sender Sender of the proof.
     * @param requestId Request id of the proof.
     * @return Proof status.
     */
    function getRequestProofStatus(
        address sender,
        uint256 requestId
    ) external view returns (RequestProofStatus memory);

    /**
     * @dev Get the proof status for the user ID and request with requestId.
     * @param userId ID of the user.
     * @param requestId Request id of the proof.
     * @return Proof status.
     */
    function getRequestProofStatusByUserId(
        uint256 userId,
        uint256 requestId
    ) external view returns (RequestProofStatus memory);
}


// ===== FILE: project/contracts/lib/GenesisUtils.sol =====
// SPDX-License-Identifier: GPL-3.0
pragma solidity 0.8.27;

import {PrimitiveTypeUtils} from "./PrimitiveTypeUtils.sol";

error ChecksumLengthRequired(uint256 length);
error IdBytesLengthRequired(uint256 length);

library GenesisUtils {
    /**
     *   @dev sum
     */
    function sum(bytes memory array) internal pure returns (uint16 s) {
        if (array.length != 29) revert ChecksumLengthRequired(29);

        for (uint256 i = 0; i < array.length; ++i) {
            s += uint16(uint8(array[i]));
        }
    }

    /**
     * @dev isGenesisState
     */
    function isGenesisState(uint256 id, uint256 idState) internal pure returns (bool) {
        bytes2 idType = getIdType(id);
        uint256 computedId = calcIdFromGenesisState(idType, idState);
        return id == computedId;
    }

    /**
     * @dev getIdType
     */
    function getIdType(uint256 id) internal pure returns (bytes2) {
        return bytes2(PrimitiveTypeUtils.uint256ToBytes(PrimitiveTypeUtils.reverseUint256(id)));
    }

    /**
     * @dev calcIdFromGenesisState
     */
    function calcIdFromGenesisState(
        bytes2 idType,
        uint256 idState
    ) internal pure returns (uint256) {
        bytes memory userStateB1 = PrimitiveTypeUtils.uint256ToBytes(
            PrimitiveTypeUtils.reverseUint256(idState)
        );

        bytes memory cutState = PrimitiveTypeUtils.slice(userStateB1, userStateB1.length - 27, 27);
        bytes memory beforeChecksum = PrimitiveTypeUtils.concat(abi.encodePacked(idType), cutState);

        uint16 checksum = PrimitiveTypeUtils.reverseUint16(sum(beforeChecksum));
        bytes memory checkSumBytes = abi.encodePacked(checksum);

        bytes memory idBytes = PrimitiveTypeUtils.concat(beforeChecksum, checkSumBytes);
        if (idBytes.length != 31) revert IdBytesLengthRequired(31);

        return PrimitiveTypeUtils.reverseUint256(PrimitiveTypeUtils.padRightToUint256(idBytes));
    }

    /**
     * @dev calcIdFromEthAddress
     */
    function calcIdFromEthAddress(bytes2 idType, address caller) internal pure returns (uint256) {
        uint256 addr = PrimitiveTypeUtils.addressToUint256(caller);

        return calcIdFromGenesisState(idType, PrimitiveTypeUtils.reverseUint256(addr));
    }
}


// ===== FILE: project/contracts/lib/PrimitiveTypeUtils.sol =====
// SPDX-License-Identifier: GPL-3.0
pragma solidity 0.8.27;

import {BytesLib} from "solidity-bytes-utils/contracts/BytesLib.sol";

error GivenInputNotAnAddressRepresentation(uint256 input);

library PrimitiveTypeUtils {
    /**
     * @dev uint256ToBytes
     */
    function uint256ToBytes(uint256 x) internal pure returns (bytes memory b) {
        b = new bytes(32);
        // solhint-disable-next-line no-inline-assembly
        assembly {
            mstore(add(b, 32), x)
        }
    }

    /**
     * @dev reverse uint256
     */
    function reverseUint256(uint256 input) internal pure returns (uint256 v) {
        v = input;

        // swap bytes
        v =
            ((v & 0xFF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00) >> 8) |
            ((v & 0x00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF) << 8);

        // swap 2-byte long pairs
        v =
            ((v & 0xFFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000) >> 16) |
            ((v & 0x0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF) << 16);

        // swap 4-byte long pairs
        v =
            ((v & 0xFFFFFFFF00000000FFFFFFFF00000000FFFFFFFF00000000FFFFFFFF00000000) >> 32) |
            ((v & 0x00000000FFFFFFFF00000000FFFFFFFF00000000FFFFFFFF00000000FFFFFFFF) << 32);

        // swap 8-byte long pairs
        v =
            ((v & 0xFFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0000000000000000) >> 64) |
            ((v & 0x0000000000000000FFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF) << 64);

        // swap 16-byte long pairs
        v = (v >> 128) | (v << 128);
    }

    /**
     * @dev reverse uint16
     */
    function reverseUint16(uint16 input) internal pure returns (uint16 v) {
        v = input;

        // swap bytes
        v = (v >> 8) | (v << 8);
    }

    /**
     * @dev reverse uint32
     */
    function reverseUint32(uint32 input) internal pure returns (uint32 v) {
        v = input;

        // swap bytes
        v = ((v & 0xFF00FF00) >> 8) | ((v & 0x00FF00FF) << 8);

        // swap 2-byte long pairs
        v = (v >> 16) | (v << 16);
    }

    /**
     * @dev compareStrings
     */
    function compareStrings(string memory a, string memory b) internal pure returns (bool) {
        if (bytes(a).length != bytes(b).length) {
            return false;
        }
        return (keccak256(abi.encodePacked((a))) == keccak256(abi.encodePacked((b))));
    }

    /**
     * @dev padRightToUint256 shift left 12 bytes
     * @param b, bytes array with max length 32, other bytes are cut. e.g. 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
     * @return value e.g 0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266000000000000000000000000
     */
    function padRightToUint256(bytes memory b) internal pure returns (uint256 value) {
        return uint256(bytes32(b));
    }

    /**
     * @dev bytesToAddress
     */
    function bytesToAddress(bytes memory bys) internal pure returns (address addr) {
        // solhint-disable-next-line no-inline-assembly
        assembly {
            addr := mload(add(bys, 20))
        }
    }

    /**
     * @dev concat
     */
    function concat(
        bytes memory preBytes,
        bytes memory postBytes
    ) internal pure returns (bytes memory) {
        return BytesLib.concat(preBytes, postBytes);
    }

    /**
     * @dev slice
     */
    function slice(
        bytes memory bys,
        uint256 start,
        uint256 length
    ) internal pure returns (bytes memory) {
        return BytesLib.slice(bys, start, length);
    }

    /**
     * @dev addressToUint256 converts address to uint256 which lower 20 bytes
     * is an address in Big Endian
     * @param _addr is ethereum address: eg.0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
     * which as 0x000000000000000000000000f39fd6e51aad88f6f4ce6ab8827279cfffb92266 converted to uint160
     * @return uint256 representation of address 1390849295786071768276380950238675083608645509734
     */
    function addressToUint256(address _addr) internal pure returns (uint256) {
        return uint256(uint160(_addr));
    }

    /**
     * @dev uint256ToAddress converts uint256 which lower 20 bytes
     * is an address in Big Endian to address
     * @param input uint256 e.g. 1390849295786071768276380950238675083608645509734
     * which as 0x000000000000000000000000f39fd6e51aad88f6f4ce6ab8827279cfffb92266 converted to address
     * @return address representation of uint256 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
     */
    function uint256ToAddress(uint256 input) internal pure returns (address) {
        if (input != uint256(uint160(input))) {
            revert GivenInputNotAnAddressRepresentation(input);
        }
        return address(uint160(input));
    }

    /**
     * @dev addressToChallenge converts address to uint256 which lower 20 bytes
     * are representation of address in LittleEndian
     * @param _addr is ethereum address: eg.0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
     * addressToBytes: 0x000000000000000000000000f39fd6e51aad88f6f4ce6ab8827279cfffb92266
     * padRightToUint256: 0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266000000000000000000000000,
     * reverseUint256 result: 0x0000000000000000000000006622b9ffcf797282b86acef4f688ad1ae5d69ff3
     * @return uint256: 583091486781463398742321306787801699791102451699
     */
    function addressToUint256LE(address _addr) internal pure returns (uint256) {
        return reverseUint256(padRightToUint256(addressToBytes(_addr)));
    }

    /**
     * @dev uint256LEtoAddress - converts uint256 which 20 lower bytes
     *      are representation of address in LE to address
     * @param input is uint256 which is created from bytes in LittleEndian:
     * eg. 583091486781463398742321306787801699791102451699
     *  or 0x0000000000000000000000006622b9ffcf797282b86acef4f688ad1ae5d69ff3
     * reverseUint256 result: 110194434039389003190498847789203126033799499726478230611233094447786700570624
     * uint256ToBytes result: 0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266000000000000000000000000
     * @return address - 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
     */
    function uint256LEToAddress(uint256 input) internal pure returns (address) {
        if (input != uint256(uint160(input))) {
            revert GivenInputNotAnAddressRepresentation(input);
        }
        return bytesToAddress(uint256ToBytes(reverseUint256(input)));
    }

    function addressToBytes(address a) internal pure returns (bytes memory) {
        return abi.encodePacked(a);
    }
}


// ===== FILE: project/contracts/lib/VerifierLib.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity 0.8.27;

import {IRequestValidator} from "../interfaces/IRequestValidator.sol";
// solhint-disable max-line-length
import {Verifier, VerifierIDIsNotValid, MultiRequestIdNotValid, GroupIdNotValid, NullifierSessionIDAlreadyExists} from "../verifiers/Verifier.sol";
import {MissingUserIDInGroupOfRequests, GroupMustHaveAtLeastTwoRequests, ResponseFieldAlreadyExists, MissingUserIDInRequest} from "../verifiers/Verifier.sol";
import {GroupIdAlreadyExists, ResponseFieldDoesNotExist, LinkIDNotTheSameForGroupedRequests, UserIDMismatch} from "../verifiers/Verifier.sol";
import {MultiRequestIdNotFound, MultiRequestIdAlreadyExists, RequestShouldNotHaveAGroup, RequestIdAlreadyExists} from "../verifiers/Verifier.sol";
import {RequestIdNotFound, RequestIdNotValid, RequestIdTypeNotValid, RequestIdUsesReservedBytes, GroupIdNotFound} from "../verifiers/Verifier.sol";
import {ProofAlreadyVerified, AuthMethodAlreadyExists, AuthMethodNotFound, ProofIsNotVerified} from "../verifiers/Verifier.sol";
import {ProofByUserIdAlreadyVerified, ProofByUserIdIsNotVerified, ResponseFieldByUserIdDoesNotExist, ResponseFieldByUserIdAlreadyExists} from "../verifiers/Verifier.sol";
// solhint-enable max-line-length
import {IVerifier} from "../interfaces/IVerifier.sol";
import {IAuthValidator} from "../interfaces/IAuthValidator.sol";

library VerifierLib {
    /// @dev Link ID field name
    string private constant LINK_ID_PROOF_FIELD_NAME = "linkID";
    /// @dev User ID field name
    string private constant USER_ID_INPUT_NAME = "userID";

    // keccak256(abi.encodePacked("userID"))
    bytes32 private constant USER_ID_FIELD_NAME_HASH =
        0xeaa28503c24395f30163098dfa9f1e1cd296dd52252064784e65d95934007382;
    // keccak256(abi.encodePacked("isEmbeddedAuthVerified"))
    bytes32 private constant IS_EMBEDDED_AUTH_VERIFIED_FIELD_NAME_HASH =
        0x77dbf4723b7e657d6a347bdf463068644740111c41d0cea27e25e1ff54c4b2db;

    /**
     * @dev Modifier to check if the request is verified
     * @param requestId The ID of the request
     * @param sender The address of the user
     * @param verification Whether request should be verified or not
     */
    modifier checkVerification(
        Verifier.VerifierStorage storage self,
        uint256 requestId,
        address sender,
        bool verification
    ) {
        if (!requestIdExists(self, requestId)) {
            revert RequestIdNotFound(requestId);
        }
        Verifier.Proof storage proof = self._proofs[requestId][sender];
        if (verification) {
            if (!proof.isVerified) {
                revert ProofIsNotVerified(requestId, sender);
            }
        } else {
            if (proof.isVerified) {
                revert ProofAlreadyVerified(requestId, sender);
            }
        }
        _;
    }

    /**
     * @dev Modifier to check if the request is verified
     * @param requestId The ID of the request
     * @param userId The ID of the user
     * @param verification Whether request should be verified or not
     */
    modifier checkVerificationByUserId(
        Verifier.VerifierStorage storage self,
        uint256 requestId,
        uint256 userId,
        bool verification
    ) {
        if (!requestIdExists(self, requestId)) {
            revert RequestIdNotFound(requestId);
        }
        Verifier.Proof storage proof = self._proofsByUserId[requestId][userId];
        if (verification) {
            if (!proof.isVerified) {
                revert ProofByUserIdIsNotVerified(requestId, userId);
            }
        } else {
            if (proof.isVerified) {
                revert ProofByUserIdAlreadyVerified(requestId, userId);
            }
        }
        _;
    }

    /**
     * @dev Modifier to check if the request exists
     */
    modifier checkRequestExistence(
        Verifier.VerifierStorage storage self,
        uint256 requestId,
        bool existence
    ) {
        if (existence) {
            if (!requestIdExists(self, requestId)) {
                revert RequestIdNotFound(requestId);
            }
        } else {
            if (requestIdExists(self, requestId)) {
                revert RequestIdAlreadyExists(requestId);
            }
        }
        _;
    }

    /**
     * @dev Modifier to check if the auth type exists
     */
    modifier checkAuthMethodExistence(
        Verifier.VerifierStorage storage self,
        string memory authMethod,
        bool existence
    ) {
        if (existence) {
            if (!authMethodExists(self, authMethod)) {
                revert AuthMethodNotFound(authMethod);
            }
        } else {
            if (authMethodExists(self, authMethod)) {
                revert AuthMethodAlreadyExists(authMethod);
            }
        }
        _;
    }

    /**
     * @dev Modifier to check if the multiRequest exists
     */
    modifier checkMultiRequestExistence(
        Verifier.VerifierStorage storage self,
        uint256 multiRequestId,
        bool existence
    ) {
        if (existence) {
            if (!multiRequestIdExists(self, multiRequestId)) {
                revert MultiRequestIdNotFound(multiRequestId);
            }
        } else {
            if (multiRequestIdExists(self, multiRequestId)) {
                revert MultiRequestIdAlreadyExists(multiRequestId);
            }
        }
        _;
    }

    function writeProofResults(
        Verifier.VerifierStorage storage self,
        uint256 requestId,
        address sender,
        uint256 userId,
        IRequestValidator.ResponseField[] memory responseFields
    ) external {
        if (!requestIdExists(self, requestId)) {
            revert RequestIdNotFound(requestId);
        }
        Verifier.Proof storage proof = self._proofs[requestId][sender];
        proof.isVerified = true;
        proof.proofEntries.push();

        string memory validatorVersion = self._requests[requestId].validator.version();

        Verifier.ProofEntry storage proofEntry = proof.proofEntries[proof.proofEntries.length - 1];
        proofEntry.validatorVersion = validatorVersion;
        proofEntry.blockTimestamp = block.timestamp;

        for (uint256 i = 0; i < responseFields.length; i++) {
            if (proofEntry.responseFieldIndexes[responseFields[i].name] != 0) {
                revert ResponseFieldAlreadyExists(requestId, sender, responseFields[i].name);
            }

            proofEntry.responseFields[responseFields[i].name] = responseFields[i].value;
            proofEntry.responseFieldNames.push(responseFields[i].name);
            // we are not using a real index defined by length-1 here but defined by just length
            // which shifts the index by 1 to avoid 0 value
            proofEntry.responseFieldIndexes[responseFields[i].name] = proofEntry
                .responseFieldNames
                .length;
        }

        if (userId != 0) {
            Verifier.Proof storage proofByUserId = self._proofsByUserId[requestId][userId];
            proofByUserId.isVerified = true;
            proofByUserId.proofEntries.push();

            Verifier.ProofEntry storage proofEntryByUserId = proofByUserId.proofEntries[
                proofByUserId.proofEntries.length - 1
            ];
            proofEntryByUserId.validatorVersion = validatorVersion;
            proofEntryByUserId.blockTimestamp = block.timestamp;

            for (uint256 i = 0; i < responseFields.length; i++) {
                if (proofEntryByUserId.responseFieldIndexes[responseFields[i].name] != 0) {
                    revert ResponseFieldByUserIdAlreadyExists(
                        requestId,
                        userId,
                        responseFields[i].name
                    );
                }

                proofEntryByUserId.responseFields[responseFields[i].name] = responseFields[i].value;
                proofEntryByUserId.responseFieldNames.push(responseFields[i].name);
                // we are not using a real index defined by length-1 here but defined by just length
                // which shifts the index by 1 to avoid 0 value
                proofEntryByUserId.responseFieldIndexes[responseFields[i].name] = proofEntryByUserId
                    .responseFieldNames
                    .length;
            }
        }
    }

    /**
     * @dev Checks if the proofs from a Multirequest submitted for a given sender and request ID are verified
     * @param multiRequestId The ID of the multiRequest
     * @param userAddress The address of the user
     * @return status The status of the multiRequest. "True" if all requests are verified, "false" otherwise
     */
    function areMultiRequestProofsVerified(
        Verifier.VerifierStorage storage self,
        uint256 multiRequestId,
        address userAddress
    ) public view checkMultiRequestExistence(self, multiRequestId, true) returns (bool) {
        // 1. Check if all requests are verified for the userAddress
        bool verified = _areMultiRequestProofsVerified(self, multiRequestId, userAddress);

        if (verified) {
            // 2. Check if all linked response fields are the same
            bool linkedResponsesOK = _checkLinkedResponseFields(self, multiRequestId, userAddress);

            if (!linkedResponsesOK) {
                verified = false;
            }
        }

        return verified;
    }

    /**
     * @dev Gets the status of the multiRequest verification
     * @param multiRequestId The ID of the multiRequest
     * @param userAddress The address of the user
     * @return status The status of the multiRequest. "True" if all requests are verified, "false" otherwise
     */
    function getMultiRequestProofsStatus(
        Verifier.VerifierStorage storage self,
        uint256 multiRequestId,
        address userAddress
    )
        public
        view
        checkMultiRequestExistence(self, multiRequestId, true)
        returns (IVerifier.RequestProofStatus[] memory)
    {
        // 1. Check if all requests statuses are true for the userAddress
        IVerifier.RequestProofStatus[] memory requestProofStatus = _getMultiRequestProofsStatus(
            self,
            multiRequestId,
            userAddress
        );

        // 2. Check if all linked response fields are the same
        bool linkedResponsesOK = _checkLinkedResponseFields(self, multiRequestId, userAddress);

        if (!linkedResponsesOK) {
            revert LinkIDNotTheSameForGroupedRequests();
        }

        return requestProofStatus;
    }

    function checkNullifierSessionIdUniqueness(
        Verifier.VerifierStorage storage self,
        IVerifier.Request calldata request
    ) external {
        uint256 nullifierSessionID = request
            .validator
            .getRequestParam(request.params, "nullifierSessionID")
            .value;
        if (nullifierSessionID != 0) {
            if (self._nullifierSessionIDs[nullifierSessionID] != 0) {
                revert NullifierSessionIDAlreadyExists(nullifierSessionID);
            }
            self._nullifierSessionIDs[nullifierSessionID] = nullifierSessionID;
        }
    }

    function checkRequestIdCorrectness(
        uint256 requestId,
        bytes calldata requestParams,
        address requestOwner
    ) external pure {
        // 1. Check prefix
        uint16 requestType = _getRequestType(requestId);
        if (requestType >= 2) {
            revert RequestIdTypeNotValid();
        }
        // 2. Check reserved bytes
        if (((requestId << 16) >> 216) > 0) {
            revert RequestIdUsesReservedBytes();
        }
        // 3. Check if requestId matches the hash of the requestParams
        // 0x0000000000000000FFFF...FF. Reserved first 8 bytes for the request Id type and future use
        // 0x00010000000000000000...00. First 2 bytes for the request Id type
        //    - 0x0000... for old request Ids with uint64
        //    - 0x0001... for new request Ids with uint256
        if (requestType == 1) {
            uint256 hashValue = uint256(keccak256(abi.encodePacked(requestParams, requestOwner)));
            uint256 expectedRequestId = (hashValue &
                0x0000000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF) +
                0x0001000000000000000000000000000000000000000000000000000000000000;
            if (requestId != expectedRequestId) {
                revert RequestIdNotValid(expectedRequestId, requestId);
            }
        }
    }

    function checkVerifierID(
        Verifier.VerifierStorage storage self,
        IVerifier.Request calldata request
    ) external view {
        uint256 requestVerifierID = request
            .validator
            .getRequestParam(request.params, "verifierID")
            .value;

        if (requestVerifierID != 0) {
            if (requestVerifierID != self._verifierID) {
                revert VerifierIDIsNotValid(requestVerifierID, self._verifierID);
            }
        }
    }

    /**
     * @dev Get the group of requests.
     * @return Group of requests.
     */
    function getGroupedRequests(
        Verifier.VerifierStorage storage self,
        uint256 groupID
    ) public view returns (IVerifier.RequestInfo[] memory) {
        IVerifier.RequestInfo[] memory requests = new IVerifier.RequestInfo[](
            self._groupedRequests[groupID].length
        );

        for (uint256 i = 0; i < self._groupedRequests[groupID].length; i++) {
            uint256 requestId = self._groupedRequests[groupID][i];
            IVerifier.RequestData storage rd = self._requests[requestId];

            requests[i] = IVerifier.RequestInfo({
                requestId: requestId,
                metadata: rd.metadata,
                validator: rd.validator,
                params: rd.params,
                creator: rd.creator
            });
        }

        return requests;
    }

    /**
     * @dev Gets a specific multiRequest by ID
     * @param multiRequestId The ID of the multiRequest
     * @return multiRequest The multiRequest data
     */
    function getMultiRequest(
        Verifier.VerifierStorage storage self,
        uint256 multiRequestId
    )
        public
        view
        checkMultiRequestExistence(self, multiRequestId, true)
        returns (IVerifier.MultiRequest memory multiRequest)
    {
        return self._multiRequests[multiRequestId];
    }

    /**
     * @dev Retrieves the value of a response field for a given request ID and user address
     * @param self The verifier storage
     * @param requestId The ID of the request
     * @param sender The address of the user
     * @param responseFieldName The name of the response field
     * @return The value of the response field
     */
    function getResponseFieldValue(
        Verifier.VerifierStorage storage self,
        uint256 requestId,
        address sender,
        string memory responseFieldName
    ) public view checkVerification(self, requestId, sender, true) returns (uint256) {
        Verifier.Proof storage proof = self._proofs[requestId][sender];
        if (
            proof.proofEntries[proof.proofEntries.length - 1].responseFieldIndexes[
                responseFieldName
            ] == 0
        ) {
            revert ResponseFieldDoesNotExist(requestId, sender, responseFieldName);
        }

        return proof.proofEntries[proof.proofEntries.length - 1].responseFields[responseFieldName];
    }

    /**
     * @dev Retrieves the value of a response field for a given request ID and user Id
     * @param self The verifier storage
     * @param requestId The ID of the request
     * @param userId The ID of the user
     * @param responseFieldName The name of the response field
     * @return The value of the response field
     */
    function getResponseFieldValueByUserId(
        Verifier.VerifierStorage storage self,
        uint256 requestId,
        uint256 userId,
        string memory responseFieldName
    ) public view checkVerificationByUserId(self, requestId, userId, true) returns (uint256) {
        Verifier.Proof storage proof = self._proofsByUserId[requestId][userId];
        if (
            proof.proofEntries[proof.proofEntries.length - 1].responseFieldIndexes[
                responseFieldName
            ] == 0
        ) {
            revert ResponseFieldByUserIdDoesNotExist(requestId, userId, responseFieldName);
        }

        return proof.proofEntries[proof.proofEntries.length - 1].responseFields[responseFieldName];
    }

    function getResponseFields(
        Verifier.VerifierStorage storage self,
        uint256 requestId,
        address sender
    )
        public
        view
        checkVerification(self, requestId, sender, true)
        returns (IRequestValidator.ResponseField[] memory)
    {
        Verifier.Proof storage proof = self._proofs[requestId][sender];
        Verifier.ProofEntry storage lastProofEntry = proof.proofEntries[
            proof.proofEntries.length - 1
        ];

        IRequestValidator.ResponseField[]
            memory responseFields = new IRequestValidator.ResponseField[](
                lastProofEntry.responseFieldNames.length
            );

        for (uint256 i = 0; i < lastProofEntry.responseFieldNames.length; i++) {
            responseFields[i] = IRequestValidator.ResponseField({
                name: lastProofEntry.responseFieldNames[i],
                value: lastProofEntry.responseFields[lastProofEntry.responseFieldNames[i]],
                rawValue: ""
            });
        }

        return responseFields;
    }

    /**
     * @dev Gets a specific request by ID
     * @param requestId The ID of the request
     * @return request The request info
     */
    function getRequest(
        Verifier.VerifierStorage storage self,
        uint256 requestId
    )
        public
        view
        checkRequestExistence(self, requestId, true)
        returns (Verifier.RequestInfo memory request)
    {
        IVerifier.RequestData storage rd = self._requests[requestId];
        return
            IVerifier.RequestInfo({
                requestId: requestId,
                metadata: rd.metadata,
                validator: rd.validator,
                params: rd.params,
                creator: rd.creator
            });
    }

    /**
     * @dev Gets the proof status of a request for a given user address
     * @param self The verifier storage
     * @param sender The address of the user
     * @param requestId The ID of the request
     * @return The proof status of the request for the user address
     */
    function getRequestProofStatus(
        Verifier.VerifierStorage storage self,
        address sender,
        uint256 requestId
    ) external view returns (IVerifier.RequestProofStatus memory) {
        Verifier.Proof storage proof = self._proofs[requestId][sender];
        if (proof.isVerified) {
            Verifier.ProofEntry storage lastProofEntry = proof.proofEntries[
                proof.proofEntries.length - 1
            ];

            return
                IVerifier.RequestProofStatus(
                    requestId,
                    true,
                    lastProofEntry.validatorVersion,
                    lastProofEntry.blockTimestamp
                );
        } else {
            return IVerifier.RequestProofStatus(requestId, false, "", 0);
        }
    }

    /**
     * @dev Gets the proof status of a request for a given user ID
     * @param self The verifier storage
     * @param userId The ID of the user
     * @param requestId The ID of the request
     * @return The proof status of the request for the user ID
     */
    function getRequestProofStatusByUserId(
        Verifier.VerifierStorage storage self,
        uint256 userId,
        uint256 requestId
    ) external view returns (IVerifier.RequestProofStatus memory) {
        Verifier.Proof storage proof = self._proofsByUserId[requestId][userId];
        if (proof.isVerified) {
            Verifier.ProofEntry storage lastProofEntry = proof.proofEntries[
                proof.proofEntries.length - 1
            ];

            return
                IVerifier.RequestProofStatus(
                    requestId,
                    true,
                    lastProofEntry.validatorVersion,
                    lastProofEntry.blockTimestamp
                );
        } else {
            return IVerifier.RequestProofStatus(requestId, false, "", 0);
        }
    }

    /**
     * @dev Checks if a request ID exists
     * @param requestId The ID of the request
     * @return Whether the request ID exists
     */
    function requestIdExists(
        Verifier.VerifierStorage storage self,
        uint256 requestId
    ) public view returns (bool) {
        return self._requests[requestId].validator != IRequestValidator(address(0));
    }

    function checkCanWriteProofResults(
        Verifier.VerifierStorage storage self,
        uint256 requestId,
        address sender
    ) public view {
        if (!requestIdExists(self, requestId)) {
            revert RequestIdNotFound(requestId);
        }
        Verifier.Proof storage proof = self._proofs[requestId][sender];

        if (proof.isVerified) {
            revert ProofAlreadyVerified(requestId, sender);
        }
    }

    function checkCanWriteProofByUserIdResults(
        Verifier.VerifierStorage storage self,
        uint256 requestId,
        uint256 userId
    ) public view {
        if (!requestIdExists(self, requestId)) {
            revert RequestIdNotFound(requestId);
        }
        Verifier.Proof storage proof = self._proofsByUserId[requestId][userId];

        if (proof.isVerified) {
            revert ProofByUserIdAlreadyVerified(requestId, userId);
        }
    }

    function checkUserIDMatch(
        uint256 userIDFromAuthResponse,
        IRequestValidator.ResponseField[] memory signals
    ) external pure {
        for (uint256 j = 0; j < signals.length; j++) {
            if (keccak256(abi.encodePacked(signals[j].name)) == USER_ID_FIELD_NAME_HASH) {
                if (userIDFromAuthResponse != signals[j].value) {
                    revert UserIDMismatch(userIDFromAuthResponse, signals[j].value);
                }
            }
        }
    }

    function userID(
        IRequestValidator.ResponseField[] memory signals
    ) public pure returns (uint256) {
        for (uint256 j = 0; j < signals.length; j++) {
            if (keccak256(abi.encodePacked(signals[j].name)) == USER_ID_FIELD_NAME_HASH) {
                return signals[j].value;
            }
        }
        return 0;
    }

    function isEmbeddedAuthVerified(
        IRequestValidator.ResponseField[] memory signals
    ) external pure returns (bool, uint256) {
        for (uint256 j = 0; j < signals.length; j++) {
            if (
                keccak256(abi.encodePacked(signals[j].name)) ==
                IS_EMBEDDED_AUTH_VERIFIED_FIELD_NAME_HASH
            ) {
                return (true, signals[j].value);
            }
        }
        return (false, 0);
    }

    /**
     * @dev Checks if an auth method exists
     * @param authMethod The auth method
     * @return Whether the auth type exists
     */
    function authMethodExists(
        Verifier.VerifierStorage storage self,
        string memory authMethod
    ) public view returns (bool) {
        return self._authMethods[authMethod].validator != IAuthValidator(address(0));
    }

    /**
     * @dev Sets an auth method
     * @param authMethod The auth method to add
     */
    function setAuthMethod(
        Verifier.VerifierStorage storage self,
        IVerifier.AuthMethod calldata authMethod
    ) external checkAuthMethodExistence(self, authMethod.authMethod, false) {
        self._authMethodsNames.push(authMethod.authMethod);
        self._authMethods[authMethod.authMethod] = Verifier.AuthMethodData({
            validator: authMethod.validator,
            params: authMethod.params,
            isActive: true
        });
    }

    /**
     * @dev Gets an auth type
     * @param authMethod The Id of the auth type to get
     * @return authMethodData The auth type data
     */
    function getAuthMethod(
        Verifier.VerifierStorage storage self,
        string calldata authMethod
    )
        external
        view
        checkAuthMethodExistence(self, authMethod, true)
        returns (Verifier.AuthMethodData memory authMethodData)
    {
        return self._authMethods[authMethod];
    }

    /**
     * @dev Enables an auth type
     * @param authMethod The auth type to enable
     */
    function enableAuthMethod(
        Verifier.VerifierStorage storage self,
        string calldata authMethod
    ) external checkAuthMethodExistence(self, authMethod, true) {
        self._authMethods[authMethod].isActive = true;
    }

    /**
     * @dev Disables an auth method
     * @param authMethod The auth method to disable
     */
    function disableAuthMethod(
        Verifier.VerifierStorage storage self,
        string calldata authMethod
    ) external checkAuthMethodExistence(self, authMethod, true) {
        self._authMethods[authMethod].isActive = false;
    }

    function checkGroupIdsAndRequestsPerGroup(
        Verifier.VerifierStorage storage self,
        IVerifier.Request[] calldata requests
    ) external {
        uint256 newGroupsCount = 0;
        Verifier.GroupInfo[] memory newGroupsInfo = new Verifier.GroupInfo[](requests.length);

        for (uint256 i = 0; i < requests.length; i++) {
            uint256 groupID = requests[i]
                .validator
                .getRequestParam(requests[i].params, "groupID")
                .value;

            if (groupID != 0) {
                (bool exists, uint256 groupIDIndex) = _getGroupIDIndex(
                    groupID,
                    newGroupsInfo,
                    newGroupsCount
                );

                if (!exists) {
                    if (groupIdExists(self, groupID)) {
                        revert GroupIdAlreadyExists(groupID);
                    }
                    self._groupIds.push(groupID);
                    self._groupedRequests[groupID].push(requests[i].requestId);

                    newGroupsInfo[newGroupsCount] = Verifier.GroupInfo({
                        id: groupID,
                        userIdInputExists: _isUserIDPublicSignalInRequest(requests[i])
                    });

                    newGroupsCount++;
                } else {
                    self._groupedRequests[groupID].push(requests[i].requestId);
                    if (_isUserIDPublicSignalInRequest(requests[i])) {
                        newGroupsInfo[groupIDIndex].userIdInputExists = true;
                    }
                }
            } else {
                // revert if standalone request is without userId public signal
                if (!_isUserIDPublicSignalInRequest(requests[i])) {
                    revert MissingUserIDInRequest(requests[i].requestId);
                }
            }
        }

        _checkGroupsRequestsInfo(self, newGroupsInfo, newGroupsCount);
    }

    /**
     * @dev Sets a multiRequest
     * @param multiRequest The multiRequest data
     */
    function setMultiRequest(
        Verifier.VerifierStorage storage self,
        IVerifier.MultiRequest calldata multiRequest,
        address sender
    ) public checkMultiRequestExistence(self, multiRequest.multiRequestId, false) {
        uint256 expectedMultiRequestId = uint256(
            keccak256(abi.encodePacked(multiRequest.requestIds, multiRequest.groupIds, sender))
        );
        if (expectedMultiRequestId != multiRequest.multiRequestId) {
            revert MultiRequestIdNotValid(expectedMultiRequestId, multiRequest.multiRequestId);
        }

        self._multiRequests[multiRequest.multiRequestId] = multiRequest;
        self._multiRequestIds.push(multiRequest.multiRequestId);

        // checks for all the requests in this multiRequest
        _checkRequestsInMultiRequest(self, multiRequest.multiRequestId);
    }

    /**
     * @dev Checks if a group ID exists
     * @param groupId The ID of the group
     * @return Whether the group ID exists
     */
    function groupIdExists(
        Verifier.VerifierStorage storage self,
        uint256 groupId
    ) public view returns (bool) {
        return self._groupedRequests[groupId].length != 0;
    }

    /**
     * @dev Checks if a multiRequest ID exists
     * @param multiRequestId The ID of the multiRequest
     * @return Whether the multiRequest ID exists
     */
    function multiRequestIdExists(
        Verifier.VerifierStorage storage self,
        uint256 multiRequestId
    ) public view returns (bool) {
        return self._multiRequests[multiRequestId].multiRequestId == multiRequestId;
    }

    function _checkRequestsInMultiRequest(
        Verifier.VerifierStorage storage self,
        uint256 multiRequestId
    ) internal view {
        uint256[] memory requestIds = self._multiRequests[multiRequestId].requestIds;
        uint256[] memory groupIds = self._multiRequests[multiRequestId].groupIds;

        // check that all the single requests doesn't have group
        for (uint256 i = 0; i < requestIds.length; i++) {
            if (!requestIdExists(self, requestIds[i])) {
                revert RequestIdNotFound(requestIds[i]);
            }
            uint256 groupID = self
                ._requests[requestIds[i]]
                .validator
                .getRequestParam(self._requests[requestIds[i]].params, "groupID")
                .value;

            if (groupID != 0) {
                revert RequestShouldNotHaveAGroup(requestIds[i]);
            }
        }

        for (uint256 i = 0; i < groupIds.length; i++) {
            if (!groupIdExists(self, groupIds[i])) {
                revert GroupIdNotFound(groupIds[i]);
            }
        }
    }

    function _getGroupIDIndex(
        uint256 groupID,
        Verifier.GroupInfo[] memory groupList,
        uint256 listCount
    ) internal pure returns (bool, uint256) {
        for (uint256 j = 0; j < listCount; j++) {
            if (groupList[j].id == groupID) {
                return (true, j);
            }
        }

        return (false, 0);
    }

    function _isUserIDPublicSignalInRequest(
        IVerifier.Request memory request
    ) internal view returns (bool) {
        bool userIDInRequests = false;
        try request.validator.inputIndexOf(USER_ID_INPUT_NAME) {
            userIDInRequests = true;
            // solhint-disable-next-line no-empty-blocks
        } catch {}

        return userIDInRequests;
    }

    function _checkGroupsRequestsInfo(
        Verifier.VerifierStorage storage self,
        Verifier.GroupInfo[] memory groupList,
        uint256 groupsCount
    ) internal view {
        for (uint256 i = 0; i < groupsCount; i++) {
            uint256 calculatedGroupIDInField = groupList[i].id &
                0x0FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
            if (calculatedGroupIDInField != groupList[i].id) {
                revert GroupIdNotValid();
            }
            if (self._groupedRequests[groupList[i].id].length < 2) {
                revert GroupMustHaveAtLeastTwoRequests(groupList[i].id);
            }
            if (groupList[i].userIdInputExists == false) {
                revert MissingUserIDInGroupOfRequests(groupList[i].id);
            }
        }
    }

    function _getRequestType(uint256 requestId) internal pure returns (uint16) {
        // 0x0000000000000000 - prefix for old uint64 requests
        // 0x0001000000000000 - prefix for keccak256 cut to fit in the remaining 192 bits
        return uint16(requestId >> 240);
    }

    function _areMultiRequestProofsVerified(
        Verifier.VerifierStorage storage self,
        uint256 multiRequestId,
        address userAddress
    ) internal view returns (bool) {
        IVerifier.MultiRequest storage multiRequest = self._multiRequests[multiRequestId];

        for (uint256 i = 0; i < multiRequest.requestIds.length; i++) {
            uint256 requestId = multiRequest.requestIds[i];

            if (!self._proofs[requestId][userAddress].isVerified) {
                return false;
            }
        }

        for (uint256 i = 0; i < multiRequest.groupIds.length; i++) {
            uint256 groupId = multiRequest.groupIds[i];

            for (uint256 j = 0; j < self._groupedRequests[groupId].length; j++) {
                uint256 requestId = self._groupedRequests[groupId][j];

                if (!self._proofs[requestId][userAddress].isVerified) {
                    return false;
                }
            }
        }

        return true;
    }

    function _getMultiRequestProofsStatus(
        Verifier.VerifierStorage storage self,
        uint256 multiRequestId,
        address userAddress
    ) internal view returns (IVerifier.RequestProofStatus[] memory) {
        IVerifier.MultiRequest storage multiRequest = self._multiRequests[multiRequestId];

        uint256 lengthGroupIds;

        if (multiRequest.groupIds.length > 0) {
            for (uint256 i = 0; i < multiRequest.groupIds.length; i++) {
                uint256 groupId = multiRequest.groupIds[i];
                lengthGroupIds += self._groupedRequests[groupId].length;
            }
        }

        IVerifier.RequestProofStatus[]
            memory requestProofStatus = new IVerifier.RequestProofStatus[](
                multiRequest.requestIds.length + lengthGroupIds
            );

        for (uint256 i = 0; i < multiRequest.requestIds.length; i++) {
            uint256 requestId = multiRequest.requestIds[i];
            Verifier.Proof storage proof = self._proofs[requestId][userAddress];

            requestProofStatus[i] = IVerifier.RequestProofStatus({
                requestId: requestId,
                isVerified: proof.isVerified,
                validatorVersion: "",
                timestamp: 0
            });

            if (proof.isVerified) {
                Verifier.ProofEntry storage lastProofEntry = proof.proofEntries[
                    proof.proofEntries.length - 1
                ];

                requestProofStatus[i].validatorVersion = lastProofEntry.validatorVersion;
                requestProofStatus[i].timestamp = lastProofEntry.blockTimestamp;
            }
        }

        for (uint256 i = 0; i < multiRequest.groupIds.length; i++) {
            uint256 groupId = multiRequest.groupIds[i];

            for (uint256 j = 0; j < self._groupedRequests[groupId].length; j++) {
                uint256 requestId = self._groupedRequests[groupId][j];
                Verifier.Proof storage proof = self._proofs[requestId][userAddress];

                requestProofStatus[multiRequest.requestIds.length + j] = IVerifier
                    .RequestProofStatus({
                        requestId: requestId,
                        isVerified: proof.isVerified,
                        validatorVersion: "",
                        timestamp: 0
                    });

                if (proof.isVerified) {
                    Verifier.ProofEntry storage lastProofEntry = proof.proofEntries[
                        proof.proofEntries.length - 1
                    ];

                    requestProofStatus[multiRequest.requestIds.length + j]
                        .validatorVersion = lastProofEntry.validatorVersion;
                    requestProofStatus[multiRequest.requestIds.length + j]
                        .timestamp = lastProofEntry.blockTimestamp;
                }
            }
        }

        return requestProofStatus;
    }

    function _checkLinkedResponseFields(
        Verifier.VerifierStorage storage self,
        uint256 multiRequestId,
        address sender
    ) internal view returns (bool) {
        for (uint256 i = 0; i < self._multiRequests[multiRequestId].groupIds.length; i++) {
            uint256 groupId = self._multiRequests[multiRequestId].groupIds[i];

            // Check linkID in the same group or requests is the same
            uint256 requestLinkID = getResponseFieldValue(
                self,
                self._groupedRequests[groupId][0],
                sender,
                LINK_ID_PROOF_FIELD_NAME
            );
            for (uint256 j = 1; j < self._groupedRequests[groupId].length; j++) {
                uint256 requestLinkIDToCompare = getResponseFieldValue(
                    self,
                    self._groupedRequests[groupId][j],
                    sender,
                    LINK_ID_PROOF_FIELD_NAME
                );
                if (requestLinkID != requestLinkIDToCompare) {
                    return false;
                }
            }
        }

        return true;
    }
}


// ===== FILE: project/contracts/verifiers/Verifier.sol =====
// SPDX-License-Identifier: GPL-3.0
pragma solidity 0.8.27;

import {VerifierLib} from "../lib/VerifierLib.sol";
import {ContextUpgradeable} from "@openzeppelin/contracts-upgradeable/utils/ContextUpgradeable.sol";
import {GenesisUtils} from "../lib/GenesisUtils.sol";
import {IAuthValidator} from "../interfaces/IAuthValidator.sol";
import {IRequestValidator} from "../interfaces/IRequestValidator.sol";
import {IState} from "../interfaces/IState.sol";
import {IVerifier} from "../interfaces/IVerifier.sol";

error AuthMethodNotFound(string authMethod);
error AuthMethodAlreadyExists(string authMethod);
error AuthMethodIsNotActive(string authMethod);
error GroupIdNotFound(uint256 groupId);
error GroupIdAlreadyExists(uint256 groupId);
error GroupMustHaveAtLeastTwoRequests(uint256 groupId);
error LinkIDNotTheSameForGroupedRequests();
error MultiRequestIdAlreadyExists(uint256 multiRequestId);
error MultiRequestIdNotFound(uint256 multiRequestId);
error MultiRequestIdNotValid(uint256 expectedMultiRequestId, uint256 multiRequestId);
error NullifierSessionIDAlreadyExists(uint256 nullifierSessionID);
error ResponseFieldDoesNotExist(uint256 requestId, address sender, string responseFieldName);
error ResponseFieldAlreadyExists(uint256 requestId, address sender, string responseFieldName);
error ResponseFieldByUserIdDoesNotExist(
    uint256 requestId,
    uint256 userId,
    string responseFieldName
);
error ResponseFieldByUserIdAlreadyExists(
    uint256 requestId,
    uint256 userId,
    string responseFieldName
);
error ProofAlreadyVerified(uint256 requestId, address sender);
error ProofIsNotVerified(uint256 requestId, address sender);
error ProofByUserIdAlreadyVerified(uint256 requestId, uint256 userId);
error ProofByUserIdIsNotVerified(uint256 requestId, uint256 userId);
error RequestIdAlreadyExists(uint256 requestId);
error RequestIdNotFound(uint256 requestId);
error RequestIdNotValid(uint256 expectedRequestId, uint256 requestId);
error RequestIdUsesReservedBytes();
error RequestIdTypeNotValid();
error RequestShouldNotHaveAGroup(uint256 requestId);
error UserIDMismatch(uint256 userIDFromAuth, uint256 userIDFromResponse);
error MissingUserIDInRequest(uint256 requestId);
error MissingUserIDInGroupOfRequests(uint256 groupId);
error UserNotAuthenticated();
error VerifierIDIsNotValid(uint256 requestVerifierID, uint256 expectedVerifierID);
error ChallengeIsInvalid();
error InvalidRequestOwner(address requestOwner, address sender);
error GroupIdNotValid();
error NoEmbeddedAuthInResponsesFound();

abstract contract Verifier is IVerifier, ContextUpgradeable {
    // keccak256(abi.encodePacked("authV2"))
    bytes32 private constant AUTHV2_METHOD_NAME_HASH =
        0x380ee2d21c7a4607d113dad9e76a0bc90f5325a136d5f0e14b6ccf849d948e25;
    // keccak256(abi.encodePacked("authV3"))
    bytes32 private constant AUTHV3_METHOD_NAME_HASH =
        0x5efa95d0461bb0b5765628b227502115d7b3ead89ff9fffbb66b8fee0fec3598;
    // keccak256(abi.encodePacked("authV3-8-32"))
    bytes32 private constant AUTHV3_8_32_METHOD_NAME_HASH =
        0x4b41d5f4907f760cdf3afde7c8d6a99e928dcddade8bec79a65c940565bc8746;
    // keccak256(abi.encodePacked("challenge"))
    bytes32 private constant CHALLENGE_FIELD_NAME_HASH =
        0x62357b294ca756256b576c5da68950c49d0d1823063551ffdcc1dad9d65a07a6;
    // keccak256(abi.encodePacked("embeddedAuth"))
    bytes32 private constant EMBEDDED_AUTH_METHOD_NAME_HASH =
        0x1705b65020b03c348229586d10f18c357cefe577bcef3ed60fad6ecd16db04ce;

    struct AuthMethodData {
        IAuthValidator validator;
        bytes params;
        bool isActive;
    }

    struct GroupInfo {
        uint256 id;
        bool userIdInputExists;
    }

    /// @custom:storage-location erc7201:iden3.storage.Verifier
    struct VerifierStorage {
        // Information about requests
        // solhint-disable-next-line
        mapping(uint256 requestId => mapping(address sender => Proof)) _proofs;
        mapping(uint256 requestId => IVerifier.RequestData) _requests;
        uint256[] _requestIds;
        IState _state;
        mapping(uint256 groupId => uint256[] requestIds) _groupedRequests;
        uint256[] _groupIds;
        // Information about multiRequests
        mapping(uint256 multiRequestId => IVerifier.MultiRequest) _multiRequests;
        uint256[] _multiRequestIds;
        // Information about auth methods and validators
        string[] _authMethodsNames;
        mapping(string authMethod => AuthMethodData) _authMethods;
        mapping(uint256 nullifierSessionID => uint256 requestId) _nullifierSessionIDs;
        // verifierID to check in requests
        uint256 _verifierID;
        mapping(uint256 requestId => mapping(uint256 userId => Proof)) _proofsByUserId;
    }

    /**
     * @dev Struct to store proof and associated data
     */
    struct Proof {
        bool isVerified;
        ProofEntry[] proofEntries;
    }

    struct ProofEntry {
        mapping(string key => uint256 inputValue) responseFields;
        string[] responseFieldNames;
        // introduce artificial shift + 1 to avoid 0 index
        mapping(string key => uint256 keyIndex) responseFieldIndexes;
        string validatorVersion;
        uint256 blockTimestamp;
        uint256[45] __gap;
    }

    // keccak256(abi.encode(uint256(keccak256("iden3.storage.Verifier")) -1 )) & ~bytes32(uint256(0xff));
    // solhint-disable-next-line const-name-snakecase
    bytes32 internal constant VerifierStorageLocation =
        0x11369addde4aae8af30dcf56fa25ad3d864848d3201d1e9197f8b4da18a51a00;

    function _getVerifierStorage() private pure returns (VerifierStorage storage $) {
        // solhint-disable-next-line no-inline-assembly
        assembly {
            $.slot := VerifierStorageLocation
        }
    }

    bytes2 internal constant VERIFIER_ID_TYPE = 0x01A1;

    /**
     * @dev Modifier to check if the request exists
     */
    modifier checkRequestExistence(uint256 requestId, bool existence) {
        if (existence) {
            if (!requestIdExists(requestId)) {
                revert RequestIdNotFound(requestId);
            }
        } else {
            if (requestIdExists(requestId)) {
                revert RequestIdAlreadyExists(requestId);
            }
        }
        _;
    }

    /**
     * @dev Modifier to check if the auth type exists
     */
    modifier checkAuthMethodExistence(string memory authMethod, bool existence) {
        if (existence) {
            if (!authMethodExists(authMethod)) {
                revert AuthMethodNotFound(authMethod);
            }
        } else {
            if (authMethodExists(authMethod)) {
                revert AuthMethodAlreadyExists(authMethod);
            }
        }
        _;
    }

    /**
     * @dev Modifier to check if the multiRequest exists
     */
    modifier checkMultiRequestExistence(uint256 multiRequestId, bool existence) {
        if (existence) {
            if (!multiRequestIdExists(multiRequestId)) {
                revert MultiRequestIdNotFound(multiRequestId);
            }
        } else {
            if (multiRequestIdExists(multiRequestId)) {
                revert MultiRequestIdAlreadyExists(multiRequestId);
            }
        }
        _;
    }

    /**
     * @dev Checks if a request ID exists
     * @param requestId The ID of the request
     * @return Whether the request ID exists
     */
    function requestIdExists(uint256 requestId) public view returns (bool) {
        return VerifierLib.requestIdExists(_getVerifierStorage(), requestId);
    }

    /**
     * @dev Checks if a group ID exists
     * @param groupId The ID of the group
     * @return Whether the group ID exists
     */
    function groupIdExists(uint256 groupId) public view returns (bool) {
        return VerifierLib.groupIdExists(_getVerifierStorage(), groupId);
    }

    /**
     * @dev Checks if a multiRequest ID exists
     * @param multiRequestId The ID of the multiRequest
     * @return Whether the multiRequest ID exists
     */
    function multiRequestIdExists(uint256 multiRequestId) public view returns (bool) {
        return VerifierLib.multiRequestIdExists(_getVerifierStorage(), multiRequestId);
    }

    /**
     * @dev Checks if an auth method exists
     * @param authMethod The auth method
     * @return Whether the auth type exists
     */
    function authMethodExists(string memory authMethod) public view returns (bool) {
        return VerifierLib.authMethodExists(_getVerifierStorage(), authMethod);
    }

    /**
     * @dev Sets different requests
     * @param requests The list of requests
     */
    function setRequests(IVerifier.Request[] calldata requests) public {
        VerifierStorage storage $ = _getVerifierStorage();
        // 1. Check first that groupIds don't exist and keep the number of requests per group.
        VerifierLib.checkGroupIdsAndRequestsPerGroup($, requests);

        // 2. Set requests checking groups and nullifierSessionID uniqueness
        for (uint256 i = 0; i < requests.length; i++) {
            VerifierLib.checkRequestIdCorrectness(
                requests[i].requestId,
                requests[i].params,
                requests[i].creator
            );

            VerifierLib.checkNullifierSessionIdUniqueness($, requests[i]);
            VerifierLib.checkVerifierID($, requests[i]);

            _setRequest(requests[i]);
        }
    }

    /**
     * @dev Gets a specific request by ID
     * @param requestId The ID of the request
     * @return request The request info
     */
    function getRequest(uint256 requestId) public view returns (RequestInfo memory request) {
        return VerifierLib.getRequest(_getVerifierStorage(), requestId);
    }

    /**
     * @dev Sets a multiRequest
     * @param multiRequest The multiRequest data
     */
    function setMultiRequest(IVerifier.MultiRequest calldata multiRequest) public virtual {
        VerifierLib.setMultiRequest(_getVerifierStorage(), multiRequest, _msgSender());
    }

    /**
     * @dev Gets a specific multiRequest by ID
     * @param multiRequestId The ID of the multiRequest
     * @return multiRequest The multiRequest data
     */
    function getMultiRequest(
        uint256 multiRequestId
    ) public view returns (IVerifier.MultiRequest memory multiRequest) {
        return VerifierLib.getMultiRequest(_getVerifierStorage(), multiRequestId);
    }

    /**
     * @dev Submits an auth response + array of responses and updates proofs status
     * - auth response with some valid auth method + responses
     * - embeddedauth auth response (no auth) + response with embedded auth proof
     * - linked proofs should be sent with auth response or embedded auth response to check userID authentication
     * @param authResponse Auth response including auth type and proof
     * @param responses The list of responses including request ID, proof and metadata for requests
     * @param crossChainProofs The list of cross chain proofs from universal resolver (oracle). This
     * includes identities and global states.
     */
    function submitResponse(
        AuthResponse memory authResponse,
        Response[] memory responses,
        bytes memory crossChainProofs
    ) public virtual {
        VerifierStorage storage $ = _getVerifierStorage();
        address sender = _msgSender();

        // 1. Process crossChainProofs
        $._state.processCrossChainProofs(crossChainProofs);

        // 2. Authenticate user and get userID
        uint256 userIDFromAuthResponse = 0;
        AuthMethodData storage authMethodData = $._authMethods[authResponse.authMethod];
        if (!authMethodData.isActive) {
            revert AuthMethodIsNotActive(authResponse.authMethod);
        }

        bytes32 authMethodNameHash = keccak256(abi.encodePacked(authResponse.authMethod));
        if (authMethodNameHash != EMBEDDED_AUTH_METHOD_NAME_HASH) {
            userIDFromAuthResponse = _processAuthMethod(
                authResponse,
                authMethodData,
                responses,
                sender
            );
        }

        // 3. Verify all the responses, check userID from signals and write proof results,
        //      emit events (existing logic)
        _checkUserIDFromResponsesAndWriteProofResults(
            responses,
            sender,
            userIDFromAuthResponse,
            authMethodNameHash
        );
    }

    /**
     * @dev Sets an auth method
     * @param authMethod The auth method to add
     */
    function setAuthMethod(IVerifier.AuthMethod calldata authMethod) public virtual {
        VerifierLib.setAuthMethod(_getVerifierStorage(), authMethod);
    }

    /**
     * @dev Disables an auth method
     * @param authMethod The auth method to disable
     */
    function disableAuthMethod(string calldata authMethod) public virtual {
        VerifierLib.disableAuthMethod(_getVerifierStorage(), authMethod);
    }

    /**
     * @dev Enables an auth type
     * @param authMethod The auth type to enable
     */
    function enableAuthMethod(string calldata authMethod) public virtual {
        VerifierLib.enableAuthMethod(_getVerifierStorage(), authMethod);
    }

    /**
     * @dev Gets an auth type
     * @param authMethod The Id of the auth type to get
     * @return authMethodData The auth type data
     */
    function getAuthMethod(
        string calldata authMethod
    ) public view returns (AuthMethodData memory authMethodData) {
        return VerifierLib.getAuthMethod(_getVerifierStorage(), authMethod);
    }

    /**
     * @dev Gets response field value
     * @param requestId Id of the request
     * @param sender Address of the user
     * @param responseFieldName Name of the response field to get
     * @return The value of the specified response field for the given sender and request ID.
     */
    function getResponseFieldValue(
        uint256 requestId,
        address sender,
        string memory responseFieldName
    ) public view returns (uint256) {
        return
            VerifierLib.getResponseFieldValue(
                _getVerifierStorage(),
                requestId,
                sender,
                responseFieldName
            );
    }

    /**
     * @dev Retrieves the value of a specific response field for a given user ID and request ID.
     * @param requestId The request ID for which to retrieve the response field value.
     * @param userId The user ID for which to retrieve the response field value.
     * @param responseFieldName The name of the response field to retrieve.
     * @return The value of the specified response field for the given user ID and request ID.
     */
    function getResponseFieldValueByUserId(
        uint256 requestId,
        uint256 userId,
        string memory responseFieldName
    ) public view returns (uint256) {
        return
            VerifierLib.getResponseFieldValueByUserId(
                _getVerifierStorage(),
                requestId,
                userId,
                responseFieldName
            );
    }

    /**
     * @dev Gets proof storage response fields
     * @param requestId Id of the request
     * @param sender Address of the user
     */
    function getResponseFields(
        uint256 requestId,
        address sender
    ) public view returns (IRequestValidator.ResponseField[] memory) {
        return VerifierLib.getResponseFields(_getVerifierStorage(), requestId, sender);
    }

    /**
     * @dev Gets the status of the multiRequest verification
     * @param multiRequestId The ID of the multiRequest
     * @param userAddress The address of the user
     * @return status The status of the multiRequest. "True" if all requests are verified, "false" otherwise
     */
    function getMultiRequestProofsStatus(
        uint256 multiRequestId,
        address userAddress
    ) public view returns (IVerifier.RequestProofStatus[] memory) {
        return
            VerifierLib.getMultiRequestProofsStatus(
                _getVerifierStorage(),
                multiRequestId,
                userAddress
            );
    }

    /**
     * @dev Checks if the proofs from a Multirequest submitted for a given sender and request ID are verified
     * @param multiRequestId The ID of the multiRequest
     * @param userAddress The address of the user
     * @return status The status of the multiRequest. "True" if all requests are verified, "false" otherwise
     */
    function areMultiRequestProofsVerified(
        uint256 multiRequestId,
        address userAddress
    ) public view returns (bool) {
        return
            VerifierLib.areMultiRequestProofsVerified(
                _getVerifierStorage(),
                multiRequestId,
                userAddress
            );
    }

    /**
     * @dev Checks if a proof from a request submitted for a given sender and request ID is verified
     * @param sender The sender's address
     * @param requestId The ID of the request
     * @return True if proof is verified
     */
    function isRequestProofVerified(
        address sender,
        uint256 requestId
    ) public view checkRequestExistence(requestId, true) returns (bool) {
        return _getVerifierStorage()._proofs[requestId][sender].isVerified;
    }

    /**
     * @dev Checks if a proof from a request submitted for a given user ID and request ID is verified
     * @param userId The ID of the user
     * @param requestId The ID of the request
     * @return True if proof is verified
     */
    function isRequestProofVerifiedByUserId(
        uint256 userId,
        uint256 requestId
    ) public view checkRequestExistence(requestId, true) returns (bool) {
        return _getVerifierStorage()._proofsByUserId[requestId][userId].isVerified;
    }

    /**
     * @dev Get the requests count.
     * @return Requests count.
     */
    function getRequestsCount() public view returns (uint256) {
        return _getVerifierStorage()._requestIds.length;
    }

    /**
     * @dev Get the group of requests count.
     * @return Group of requests count.
     */
    function getGroupsCount() public view returns (uint256) {
        return _getVerifierStorage()._groupIds.length;
    }

    /**
     * @dev Get the group of requests.
     * @return Group of requests.
     */
    function getGroupedRequests(
        uint256 groupID
    ) public view returns (IVerifier.RequestInfo[] memory) {
        return VerifierLib.getGroupedRequests(_getVerifierStorage(), groupID);
    }

    /**
     * @dev Gets the address of the state contract linked to the verifier
     * @return address State contract address
     */
    function getStateAddress() public view virtual returns (address) {
        return address(_getVerifierStorage()._state);
    }

    /**
     * @dev Gets the verifierID of the verifier contract
     * @return uint256 verifierID of the verifier contract
     */
    function getVerifierID() public view virtual returns (uint256) {
        return _getVerifierStorage()._verifierID;
    }

    /**
     * @dev Checks the proof status for a given user and request ID
     * @param sender The sender's address
     * @param requestId The ID of the ZKP request
     * @return The proof status structure
     */
    function getRequestProofStatus(
        address sender,
        uint256 requestId
    )
        public
        view
        checkRequestExistence(requestId, true)
        returns (IVerifier.RequestProofStatus memory)
    {
        return VerifierLib.getRequestProofStatus(_getVerifierStorage(), sender, requestId);
    }

    /**
     * @dev Gets the proof status for a given user ID and request ID
     * @param userId The user ID for which to get the proof status
     * @param requestId The request ID for which to get the proof status
     * @return The proof status for the given user ID and request ID
     */
    function getRequestProofStatusByUserId(
        uint256 userId,
        uint256 requestId
    )
        public
        view
        checkRequestExistence(requestId, true)
        returns (IVerifier.RequestProofStatus memory)
    {
        return VerifierLib.getRequestProofStatusByUserId(_getVerifierStorage(), userId, requestId);
    }

    function _setState(IState state) internal {
        _getVerifierStorage()._state = state;
    }

    // solhint-disable-next-line func-name-mixedcase
    function __Verifier_init(IState state) internal onlyInitializing {
        __Verifier_init_unchained(state);
    }

    // solhint-disable-next-line func-name-mixedcase
    function __Verifier_init_unchained(IState state) internal onlyInitializing {
        _setState(state);
        // initial calculation of verifierID from contract address and verifier id type defined
        uint256 calculatedVerifierID = GenesisUtils.calcIdFromEthAddress(
            VERIFIER_ID_TYPE,
            address(this)
        );
        _setVerifierID(calculatedVerifierID);
    }

    function _setVerifierID(uint256 verifierID) internal {
        VerifierStorage storage s = _getVerifierStorage();
        s._verifierID = verifierID;
    }

    function _setRequest(
        Request calldata request
    ) internal virtual checkRequestExistence(request.requestId, false) {
        _checkRequestOwner(request);

        VerifierStorage storage s = _getVerifierStorage();

        s._requests[request.requestId] = IVerifier.RequestData({
            metadata: request.metadata,
            validator: request.validator,
            params: request.params,
            creator: _msgSender()
        });
        s._requestIds.push(request.requestId);
    }

    function _checkRequestOwner(Request calldata request) internal virtual {
        if (request.creator != _msgSender()) {
            revert InvalidRequestOwner(request.creator, _msgSender());
        }
    }

    /**
     * @dev Updates a request
     * @param request The request data
     */
    function _updateRequest(
        IVerifier.Request calldata request
    ) internal checkRequestExistence(request.requestId, true) {
        VerifierStorage storage s = _getVerifierStorage();

        s._requests[request.requestId] = IVerifier.RequestData({
            metadata: request.metadata,
            validator: request.validator,
            params: request.params,
            creator: request.creator
        });
    }

    function _checkCanWriteProofResults(uint256 requestId, address sender) internal view virtual {
        VerifierLib.checkCanWriteProofResults(_getVerifierStorage(), requestId, sender);
    }

    function _checkCanWriteProofByUserIdResults(
        uint256 requestId,
        uint256 userId
    ) internal view virtual {
        VerifierLib.checkCanWriteProofByUserIdResults(_getVerifierStorage(), requestId, userId);
    }

    function _getRequestIfCanBeVerified(
        uint256 requestId
    )
        internal
        view
        virtual
        checkRequestExistence(requestId, true)
        returns (IVerifier.RequestData storage)
    {
        return _getVerifierStorage()._requests[requestId];
    }

    function _checkUserIDFromResponsesAndWriteProofResults(
        Response[] memory responses,
        address sender,
        uint256 userIDFromAuthResponse,
        bytes32 authMethodNameHash
    ) internal {
        for (uint256 i = 0; i < responses.length; i++) {
            IVerifier.Response memory response = responses[i];
            IVerifier.RequestData storage request = _getRequestIfCanBeVerified(response.requestId);

            IRequestValidator.ResponseField[] memory responseFields = request.validator.verify(
                sender,
                response.proof,
                request.params,
                response.metadata
            );

            if (authMethodNameHash == EMBEDDED_AUTH_METHOD_NAME_HASH) {
                // Check isEmbeddedAuthVerified response field is present in the response fields from the validator
                // verification
                // If it's present it should be equal to 1 because we are checking embeddedAuth auth method in
                // validators that support it
                // For linkMultiQueryValidator we don't have this response field because it's always linked
                // to other responses that will have this embedded auth verified field
                (bool hasEmbeddedAuthVerified, uint256 embeddedAuthVerifiedValue) = VerifierLib
                    .isEmbeddedAuthVerified(responseFields);

                if (hasEmbeddedAuthVerified) {
                    if (embeddedAuthVerifiedValue != 1) {
                        revert NoEmbeddedAuthInResponsesFound();
                    }
                    if (userIDFromAuthResponse == 0) {
                        // If embedded auth method is used, we can use first userID from responses
                        userIDFromAuthResponse = VerifierLib.userID(responseFields);
                    }
                }
                if (userIDFromAuthResponse == 0) {
                    revert MissingUserIDInRequest(response.requestId);
                }
            }
            // Check if userID from authResponse is the same as the one in the responseFields
            VerifierLib.checkUserIDMatch(userIDFromAuthResponse, responseFields);

            _writeProofResults(response.requestId, sender, userIDFromAuthResponse, responseFields);
        }
    }

    /**
     * @dev Writes proof results.
     * @param requestId The request ID of the proof
     * @param sender The address of the sender of the proof
     * @param responseFields The array of response fields of the proof
     */
    function _writeProofResults(
        uint256 requestId,
        address sender,
        uint256 userId,
        IRequestValidator.ResponseField[] memory responseFields
    ) internal {
        _checkCanWriteProofResults(requestId, sender);

        if (userId != 0) {
            _checkCanWriteProofByUserIdResults(requestId, userId);
        }

        return
            VerifierLib.writeProofResults(
                _getVerifierStorage(),
                requestId,
                sender,
                userId,
                responseFields
            );
    }

    function _processAuthMethod(
        AuthResponse memory authResponse,
        AuthMethodData memory authMethodData,
        Response[] memory responses,
        address sender
    ) internal returns (uint256) {
        IAuthValidator.AuthResponseField[] memory authResponseFields;
        uint256 userIDFromAuthResponse = 0;
        (userIDFromAuthResponse, authResponseFields) = authMethodData.validator.verify(
            sender,
            authResponse.proof,
            authMethodData.params
        );

        if (
            keccak256(abi.encodePacked(authResponse.authMethod)) == AUTHV2_METHOD_NAME_HASH ||
            keccak256(abi.encodePacked(authResponse.authMethod)) == AUTHV3_METHOD_NAME_HASH ||
            keccak256(abi.encodePacked(authResponse.authMethod)) == AUTHV3_8_32_METHOD_NAME_HASH
        ) {
            if (
                authResponseFields.length > 0 &&
                keccak256(abi.encodePacked(authResponseFields[0].name)) == CHALLENGE_FIELD_NAME_HASH
            ) {
                bytes32 expectedNonce = keccak256(abi.encode(sender, responses)) &
                    0x0FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
                if (expectedNonce != bytes32(authResponseFields[0].value)) {
                    revert ChallengeIsInvalid();
                }
            }
        }

        if (userIDFromAuthResponse == 0) {
            revert UserNotAuthenticated();
        }
        return userIDFromAuthResponse;
    }
}
