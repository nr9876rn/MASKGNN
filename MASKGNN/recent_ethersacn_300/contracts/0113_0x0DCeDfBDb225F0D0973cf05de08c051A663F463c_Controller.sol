// ===== FILE: lib/diamond-pau/src/Controller.sol =====
// SPDX-License-Identifier: AGPL-3.0-or-later
pragma solidity ^0.8.34;

import {
    ReentrancyGuardUpgradeable
} from "../lib/oz-upgradeable/contracts/utils/ReentrancyGuardUpgradeable.sol";

import {
    EnumerableSet
} from "../lib/openzeppelin-contracts/contracts/utils/structs/EnumerableSet.sol";

import { IAccessControls } from "./interfaces/IAccessControls.sol";
import { IBeacon }         from "./interfaces/IBeacon.sol";
import { IController }     from "./interfaces/IController.sol";

import { ControllerSharedStorage } from "./ControllerSharedStorage.sol";

contract Controller is IController, ControllerSharedStorage, ReentrancyGuardUpgradeable {

    using EnumerableSet for EnumerableSet.Bytes32Set;

    /**********************************************************************************************/
    /*** Controller Storage Domain                                                              ***/
    /**********************************************************************************************/

    /// @custom:storage-location erc7201:sky.pau.storage.Controller
    struct ControllerStorage {
        EnumerableSet.Bytes32Set integrationIds;
        mapping (bytes32 integrationId => Config config)     configs;
        mapping (bytes4  callSelector  => Dispatch dispatch) dispatches;
    }

    // keccak256(abi.encode(uint256(keccak256("sky.pau.storage.Controller")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 internal constant CONTROLLER_STORAGE_LOCATION =
        0xee25394e09bdf9f095ffaf6289395c59de06e33ff54692b0774d5012253c4d00;

    function _getControllerStorage() internal pure returns (ControllerStorage storage $) {
        assembly {
            $.slot := CONTROLLER_STORAGE_LOCATION
        }
    }

    /**********************************************************************************************/
    /*** Constants                                                                              ***/
    /**********************************************************************************************/

    bytes32 internal constant _DEFAULT_ADMIN_ROLE = 0x00;

    /**********************************************************************************************/
    /*** Declarations                                                                           ***/
    /**********************************************************************************************/

    address public immutable override beacon;

    /**********************************************************************************************/
    /*** Modifiers                                                                              ***/
    /**********************************************************************************************/

    modifier onlyAdmin() {
        _revertIfNotAdmin();
        _;
    }

    /**********************************************************************************************/
    /*** Constructor                                                                            ***/
    /**********************************************************************************************/

    constructor(address accessControls_, address beacon_, address proxy_, address rateLimits_)
        initializer
    {
        require(accessControls_ != address(0), ZeroAccessControls());
        require(beacon_         != address(0), ZeroBeacon());
        require(proxy_          != address(0), ZeroProxy());
        require(rateLimits_     != address(0), ZeroRateLimits());

        __ReentrancyGuard_init();

        SharedControllerStorage storage $ = _getSharedControllerStorage();

        $.accessControls = accessControls_;
        $.proxy          = proxy_;
        $.rateLimits     = rateLimits_;

        beacon = beacon_;
    }

    /**********************************************************************************************/
    /*** External Interactive Admin Functions                                                   ***/
    /**********************************************************************************************/

    function updateIntegrations(bytes32[] calldata ids) external override nonReentrant onlyAdmin {
        require(ids.length > 0, EmptyArray());

        Config[] memory configs = IBeacon(beacon).getConfigs(ids);

        ControllerStorage storage $ = _getControllerStorage();

        // Iterate over all integrationIds and set/overwrite each integration config
        for (uint256 i = 0; i < ids.length; ++i) {
            bytes32 id = ids[i];

            Config memory config = configs[i];

            require(config.facet != address(0),   IntegrationNotFound(id));
            require(config.facet.code.length > 0, EmptyFacet());
            require(config.wires.length > 0,      EmptyArray());

            if (!$.integrationIds.add(id)) {
                // Remove the existing config and dispatches for this integration.
                _deleteConfigAndDispatches(id);
            }

            _setConfigAndDispatches(id, config);

            emit IntegrationSet(id, config);
        }
    }

    function removeIntegrations(bytes32[] calldata ids) external override nonReentrant onlyAdmin {
        require(ids.length > 0, EmptyArray());

        ControllerStorage storage $ = _getControllerStorage();

        for (uint256 i = 0; i < ids.length; ++i) {
            bytes32 id = ids[i];

            require($.integrationIds.remove(id), IntegrationNotFound(id));

            _deleteConfigAndDispatches(id);

            emit IntegrationRemoved(id);
        }
    }

    /**********************************************************************************************/
    /*** External Variable Getters                                                              ***/
    /**********************************************************************************************/

    function accessControls() external view override returns (address) {
        return _getSharedControllerStorage().accessControls;
    }

    function integrations() external view override returns (Integration[] memory integrations_) {
        ControllerStorage storage $ = _getControllerStorage();

        uint256 integrationCount = $.integrationIds.length();

        integrations_ = new Integration[](integrationCount);

        for (uint256 i = 0; i < integrationCount; ++i) {
            bytes32 id = $.integrationIds.at(i);

            integrations_[i] = Integration(id, $.configs[id]);
        }
    }

    function proxy() external view override returns (address) {
        return _getSharedControllerStorage().proxy;
    }

    function rateLimits() external view override returns (address) {
        return _getSharedControllerStorage().rateLimits;
    }

    /**********************************************************************************************/
    /*** External View/Pure Functions                                                           ***/
    /**********************************************************************************************/

    function getConfig(bytes32 integrationId) external view override returns (Config memory) {
        return _getControllerStorage().configs[integrationId];
    }

    function getConfigs(bytes32[] calldata integrationIds)
        external
        view
        override
        returns (Config[] memory integrationsConfig_)
    {
        integrationsConfig_ = new Config[](integrationIds.length);

        ControllerStorage storage $ = _getControllerStorage();

        for (uint256 i = 0; i < integrationIds.length; ++i) {
            integrationsConfig_[i] = $.configs[integrationIds[i]];
        }
    }

    function getDispatch(bytes4 callSelector)
        external
        view
        override
        returns (Dispatch memory dispatch)
    {
        return _getControllerStorage().dispatches[callSelector];
    }

    function getDispatches(bytes4[] calldata callSelectors)
        external
        view
        override
        returns (Dispatch[] memory dispatches)
    {
        dispatches = new Dispatch[](callSelectors.length);

        ControllerStorage storage $ = _getControllerStorage();

        for (uint256 i = 0; i < callSelectors.length; ++i) {
            dispatches[i] = $.dispatches[callSelectors[i]];
        }
    }

    /**********************************************************************************************/
    /*** Fallback Functions                                                                     ***/
    /**********************************************************************************************/

    fallback() external payable {
        require(msg.data.length >= 4, InvalidCallDataLength(msg.data.length));

        Dispatch storage dispatch = _getControllerStorage().dispatches[msg.sig];

        address facet = dispatch.facet;

        require(facet != address(0), CallSelectorNotWired(msg.sig));

        // Replace the incoming selector with the delegate selector.
        ( bool success, bytes memory returnData ) = facet.delegatecall(
            abi.encodePacked(dispatch.delegateSelector, msg.data[4:])
        );

        // Forward return data as-is (not possible without assembly in a fallback).
        // slither-disable-next-line assembly
        assembly {
            switch success
            case 0  { revert(add(returnData, 0x20), mload(returnData)) }
            default { return(add(returnData, 0x20), mload(returnData)) }
        }
    }

    /**********************************************************************************************/
    /*** Internal Interactive Functions                                                         ***/
    /**********************************************************************************************/

    function _deleteConfigAndDispatches(bytes32 integrationId) internal {
        ControllerStorage storage $     = _getControllerStorage();
        Wire[]            storage wires = $.configs[integrationId].wires;

        for (uint256 i = wires.length; i > 0;) {
            delete $.dispatches[wires[--i].callSelector];
            wires.pop();
        }

        delete $.configs[integrationId];
    }

    function _setConfigAndDispatches(bytes32 id, Config memory config) internal {
        ControllerStorage storage $            = _getControllerStorage();
        Config            storage storedConfig = $.configs[id];

        storedConfig.facet = config.facet;

        Wire[] memory wires = config.wires;

        for (uint256 i = 0; i < wires.length; ++i) {
            bytes4 callSelector     = wires[i].callSelector;
            bytes4 delegateSelector = wires[i].delegateSelector;

            require(
                $.dispatches[callSelector].facet == address(0),
                CallSelectorAlreadyWired(callSelector)
            );

            storedConfig.wires.push(wires[i]);

            $.dispatches[callSelector] = Dispatch(config.facet, delegateSelector);
        }
    }

    /**********************************************************************************************/
    /*** Internal View/Pure Functions                                                           ***/
    /**********************************************************************************************/

    function _revertIfNotAdmin() internal view {
        require(
            IAccessControls(_getSharedControllerStorage().accessControls).hasRole(
                _DEFAULT_ADMIN_ROLE,
                msg.sender
            ),
            NotAdmin(msg.sender)
        );
    }

}


// ===== FILE: lib/diamond-pau/lib/oz-upgradeable/contracts/utils/ReentrancyGuardUpgradeable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (utils/ReentrancyGuard.sol)

pragma solidity ^0.8.20;
import {Initializable} from "../proxy/utils/Initializable.sol";

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
abstract contract ReentrancyGuardUpgradeable is Initializable {
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

    /// @custom:storage-location erc7201:openzeppelin.storage.ReentrancyGuard
    struct ReentrancyGuardStorage {
        uint256 _status;
    }

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.ReentrancyGuard")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant ReentrancyGuardStorageLocation = 0x9b779b17422d0df92223018b32b4d1fa46e071723d6817e2486d003becc55f00;

    function _getReentrancyGuardStorage() private pure returns (ReentrancyGuardStorage storage $) {
        assembly {
            $.slot := ReentrancyGuardStorageLocation
        }
    }

    /**
     * @dev Unauthorized reentrant call.
     */
    error ReentrancyGuardReentrantCall();

    function __ReentrancyGuard_init() internal onlyInitializing {
        __ReentrancyGuard_init_unchained();
    }

    function __ReentrancyGuard_init_unchained() internal onlyInitializing {
        ReentrancyGuardStorage storage $ = _getReentrancyGuardStorage();
        $._status = NOT_ENTERED;
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
        ReentrancyGuardStorage storage $ = _getReentrancyGuardStorage();
        // On the first call to nonReentrant, _status will be NOT_ENTERED
        if ($._status == ENTERED) {
            revert ReentrancyGuardReentrantCall();
        }

        // Any calls to nonReentrant after this point will fail
        $._status = ENTERED;
    }

    function _nonReentrantAfter() private {
        ReentrancyGuardStorage storage $ = _getReentrancyGuardStorage();
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        $._status = NOT_ENTERED;
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        ReentrancyGuardStorage storage $ = _getReentrancyGuardStorage();
        return $._status == ENTERED;
    }
}


// ===== FILE: lib/diamond-pau/lib/openzeppelin-contracts/contracts/utils/structs/EnumerableSet.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (utils/structs/EnumerableSet.sol)
// This file was procedurally generated from scripts/generate/templates/EnumerableSet.js.

pragma solidity ^0.8.20;

/**
 * @dev Library for managing
 * https://en.wikipedia.org/wiki/Set_(abstract_data_type)[sets] of primitive
 * types.
 *
 * Sets have the following properties:
 *
 * - Elements are added, removed, and checked for existence in constant time
 * (O(1)).
 * - Elements are enumerated in O(n). No guarantees are made on the ordering.
 *
 * ```solidity
 * contract Example {
 *     // Add the library methods
 *     using EnumerableSet for EnumerableSet.AddressSet;
 *
 *     // Declare a set state variable
 *     EnumerableSet.AddressSet private mySet;
 * }
 * ```
 *
 * As of v3.3.0, sets of type `bytes32` (`Bytes32Set`), `address` (`AddressSet`)
 * and `uint256` (`UintSet`) are supported.
 *
 * [WARNING]
 * ====
 * Trying to delete such a structure from storage will likely result in data corruption, rendering the structure
 * unusable.
 * See https://github.com/ethereum/solidity/pull/11843[ethereum/solidity#11843] for more info.
 *
 * In order to clean an EnumerableSet, you can either remove all elements one by one or create a fresh instance using an
 * array of EnumerableSet.
 * ====
 */
library EnumerableSet {
    // To implement this library for multiple types with as little code
    // repetition as possible, we write it in terms of a generic Set type with
    // bytes32 values.
    // The Set implementation uses private functions, and user-facing
    // implementations (such as AddressSet) are just wrappers around the
    // underlying Set.
    // This means that we can only create new EnumerableSets for types that fit
    // in bytes32.

    struct Set {
        // Storage of set values
        bytes32[] _values;
        // Position is the index of the value in the `values` array plus 1.
        // Position 0 is used to mean a value is not in the set.
        mapping(bytes32 value => uint256) _positions;
    }

    /**
     * @dev Add a value to a set. O(1).
     *
     * Returns true if the value was added to the set, that is if it was not
     * already present.
     */
    function _add(Set storage set, bytes32 value) private returns (bool) {
        if (!_contains(set, value)) {
            set._values.push(value);
            // The value is stored at length-1, but we add 1 to all indexes
            // and use 0 as a sentinel value
            set._positions[value] = set._values.length;
            return true;
        } else {
            return false;
        }
    }

    /**
     * @dev Removes a value from a set. O(1).
     *
     * Returns true if the value was removed from the set, that is if it was
     * present.
     */
    function _remove(Set storage set, bytes32 value) private returns (bool) {
        // We cache the value's position to prevent multiple reads from the same storage slot
        uint256 position = set._positions[value];

        if (position != 0) {
            // Equivalent to contains(set, value)
            // To delete an element from the _values array in O(1), we swap the element to delete with the last one in
            // the array, and then remove the last element (sometimes called as 'swap and pop').
            // This modifies the order of the array, as noted in {at}.

            uint256 valueIndex = position - 1;
            uint256 lastIndex = set._values.length - 1;

            if (valueIndex != lastIndex) {
                bytes32 lastValue = set._values[lastIndex];

                // Move the lastValue to the index where the value to delete is
                set._values[valueIndex] = lastValue;
                // Update the tracked position of the lastValue (that was just moved)
                set._positions[lastValue] = position;
            }

            // Delete the slot where the moved value was stored
            set._values.pop();

            // Delete the tracked position for the deleted slot
            delete set._positions[value];

            return true;
        } else {
            return false;
        }
    }

    /**
     * @dev Returns true if the value is in the set. O(1).
     */
    function _contains(Set storage set, bytes32 value) private view returns (bool) {
        return set._positions[value] != 0;
    }

    /**
     * @dev Returns the number of values on the set. O(1).
     */
    function _length(Set storage set) private view returns (uint256) {
        return set._values.length;
    }

    /**
     * @dev Returns the value stored at position `index` in the set. O(1).
     *
     * Note that there are no guarantees on the ordering of values inside the
     * array, and it may change when more values are added or removed.
     *
     * Requirements:
     *
     * - `index` must be strictly less than {length}.
     */
    function _at(Set storage set, uint256 index) private view returns (bytes32) {
        return set._values[index];
    }

    /**
     * @dev Return the entire set in an array
     *
     * WARNING: This operation will copy the entire storage to memory, which can be quite expensive. This is designed
     * to mostly be used by view accessors that are queried without any gas fees. Developers should keep in mind that
     * this function has an unbounded cost, and using it as part of a state-changing function may render the function
     * uncallable if the set grows to a point where copying to memory consumes too much gas to fit in a block.
     */
    function _values(Set storage set) private view returns (bytes32[] memory) {
        return set._values;
    }

    // Bytes32Set

    struct Bytes32Set {
        Set _inner;
    }

    /**
     * @dev Add a value to a set. O(1).
     *
     * Returns true if the value was added to the set, that is if it was not
     * already present.
     */
    function add(Bytes32Set storage set, bytes32 value) internal returns (bool) {
        return _add(set._inner, value);
    }

    /**
     * @dev Removes a value from a set. O(1).
     *
     * Returns true if the value was removed from the set, that is if it was
     * present.
     */
    function remove(Bytes32Set storage set, bytes32 value) internal returns (bool) {
        return _remove(set._inner, value);
    }

    /**
     * @dev Returns true if the value is in the set. O(1).
     */
    function contains(Bytes32Set storage set, bytes32 value) internal view returns (bool) {
        return _contains(set._inner, value);
    }

    /**
     * @dev Returns the number of values in the set. O(1).
     */
    function length(Bytes32Set storage set) internal view returns (uint256) {
        return _length(set._inner);
    }

    /**
     * @dev Returns the value stored at position `index` in the set. O(1).
     *
     * Note that there are no guarantees on the ordering of values inside the
     * array, and it may change when more values are added or removed.
     *
     * Requirements:
     *
     * - `index` must be strictly less than {length}.
     */
    function at(Bytes32Set storage set, uint256 index) internal view returns (bytes32) {
        return _at(set._inner, index);
    }

    /**
     * @dev Return the entire set in an array
     *
     * WARNING: This operation will copy the entire storage to memory, which can be quite expensive. This is designed
     * to mostly be used by view accessors that are queried without any gas fees. Developers should keep in mind that
     * this function has an unbounded cost, and using it as part of a state-changing function may render the function
     * uncallable if the set grows to a point where copying to memory consumes too much gas to fit in a block.
     */
    function values(Bytes32Set storage set) internal view returns (bytes32[] memory) {
        bytes32[] memory store = _values(set._inner);
        bytes32[] memory result;

        /// @solidity memory-safe-assembly
        assembly {
            result := store
        }

        return result;
    }

    // AddressSet

    struct AddressSet {
        Set _inner;
    }

    /**
     * @dev Add a value to a set. O(1).
     *
     * Returns true if the value was added to the set, that is if it was not
     * already present.
     */
    function add(AddressSet storage set, address value) internal returns (bool) {
        return _add(set._inner, bytes32(uint256(uint160(value))));
    }

    /**
     * @dev Removes a value from a set. O(1).
     *
     * Returns true if the value was removed from the set, that is if it was
     * present.
     */
    function remove(AddressSet storage set, address value) internal returns (bool) {
        return _remove(set._inner, bytes32(uint256(uint160(value))));
    }

    /**
     * @dev Returns true if the value is in the set. O(1).
     */
    function contains(AddressSet storage set, address value) internal view returns (bool) {
        return _contains(set._inner, bytes32(uint256(uint160(value))));
    }

    /**
     * @dev Returns the number of values in the set. O(1).
     */
    function length(AddressSet storage set) internal view returns (uint256) {
        return _length(set._inner);
    }

    /**
     * @dev Returns the value stored at position `index` in the set. O(1).
     *
     * Note that there are no guarantees on the ordering of values inside the
     * array, and it may change when more values are added or removed.
     *
     * Requirements:
     *
     * - `index` must be strictly less than {length}.
     */
    function at(AddressSet storage set, uint256 index) internal view returns (address) {
        return address(uint160(uint256(_at(set._inner, index))));
    }

    /**
     * @dev Return the entire set in an array
     *
     * WARNING: This operation will copy the entire storage to memory, which can be quite expensive. This is designed
     * to mostly be used by view accessors that are queried without any gas fees. Developers should keep in mind that
     * this function has an unbounded cost, and using it as part of a state-changing function may render the function
     * uncallable if the set grows to a point where copying to memory consumes too much gas to fit in a block.
     */
    function values(AddressSet storage set) internal view returns (address[] memory) {
        bytes32[] memory store = _values(set._inner);
        address[] memory result;

        /// @solidity memory-safe-assembly
        assembly {
            result := store
        }

        return result;
    }

    // UintSet

    struct UintSet {
        Set _inner;
    }

    /**
     * @dev Add a value to a set. O(1).
     *
     * Returns true if the value was added to the set, that is if it was not
     * already present.
     */
    function add(UintSet storage set, uint256 value) internal returns (bool) {
        return _add(set._inner, bytes32(value));
    }

    /**
     * @dev Removes a value from a set. O(1).
     *
     * Returns true if the value was removed from the set, that is if it was
     * present.
     */
    function remove(UintSet storage set, uint256 value) internal returns (bool) {
        return _remove(set._inner, bytes32(value));
    }

    /**
     * @dev Returns true if the value is in the set. O(1).
     */
    function contains(UintSet storage set, uint256 value) internal view returns (bool) {
        return _contains(set._inner, bytes32(value));
    }

    /**
     * @dev Returns the number of values in the set. O(1).
     */
    function length(UintSet storage set) internal view returns (uint256) {
        return _length(set._inner);
    }

    /**
     * @dev Returns the value stored at position `index` in the set. O(1).
     *
     * Note that there are no guarantees on the ordering of values inside the
     * array, and it may change when more values are added or removed.
     *
     * Requirements:
     *
     * - `index` must be strictly less than {length}.
     */
    function at(UintSet storage set, uint256 index) internal view returns (uint256) {
        return uint256(_at(set._inner, index));
    }

    /**
     * @dev Return the entire set in an array
     *
     * WARNING: This operation will copy the entire storage to memory, which can be quite expensive. This is designed
     * to mostly be used by view accessors that are queried without any gas fees. Developers should keep in mind that
     * this function has an unbounded cost, and using it as part of a state-changing function may render the function
     * uncallable if the set grows to a point where copying to memory consumes too much gas to fit in a block.
     */
    function values(UintSet storage set) internal view returns (uint256[] memory) {
        bytes32[] memory store = _values(set._inner);
        uint256[] memory result;

        /// @solidity memory-safe-assembly
        assembly {
            result := store
        }

        return result;
    }
}


// ===== FILE: lib/diamond-pau/src/interfaces/IAccessControls.sol =====
// SPDX-License-Identifier: AGPL-3.0-or-later
pragma solidity ^0.8.34;

import {
    IAccessControlEnumerable
} from "../../lib/openzeppelin-contracts/contracts/access/extensions/IAccessControlEnumerable.sol";

/**
 * @title  IAccessControls
 * @notice Role-based access control interface for PAU system, extending OpenZeppelin's
 *         AccessControlEnumerable.
 */
interface IAccessControls is IAccessControlEnumerable {

    /**********************************************************************************************/
    /*** Custom Errors                                                                          ***/
    /**********************************************************************************************/

    /// @notice Thrown when the admin is the zero address.
    error ZeroAdmin();

    /**********************************************************************************************/
    /*** Interactive Admin Functions                                                            ***/
    /**********************************************************************************************/

    /**
     * @notice Sets the admin role for a given role.
     * @param role      The role to set the admin for.
     * @param adminRole The admin role for the given role.
     */
    function setRoleAdmin(bytes32 role, bytes32 adminRole) external;

    /**********************************************************************************************/
    /*** View/Pure Functions                                                                    ***/
    /**********************************************************************************************/

    /**
     * @notice Returns true if the contract supports the given interface.
     * @param  interfaceId The 4-byte interface identifier (ERC-165).
     * @return isSupported True if the interface is supported, false otherwise.
     */
    function supportsInterface(bytes4 interfaceId) external view returns (bool);

}


// ===== FILE: lib/diamond-pau/src/interfaces/IBeacon.sol =====
// SPDX-License-Identifier: AGPL-3.0-or-later
pragma solidity ^0.8.34;

import {
    IAccessControlEnumerable
} from "../../lib/openzeppelin-contracts/contracts/access/extensions/IAccessControlEnumerable.sol";

import { IEnumerableIntegrations } from "./IEnumerableIntegrations.sol";

/**
 * @title  IBeacon
 * @notice Configuration registry for the PAU system. Stores integration configs (facet
 *         address and selector wiring) that Controllers sync from to build their dispatch tables.
 */
interface IBeacon is IAccessControlEnumerable, IEnumerableIntegrations {

    /**********************************************************************************************/
    /*** Custom Errors                                                                          ***/
    /**********************************************************************************************/

    /// @notice Thrown when the call selector is hardcoded.
    error CallSelectorHardcoded(bytes4 callSelector);

    /// @notice Thrown when an argument array is empty.
    error EmptyArray();

    /// @notice Thrown when the admin is the zero address.
    error ZeroAdmin();

    /// @notice Thrown when the facet is the zero address.
    error ZeroFacet();

    /**********************************************************************************************/
    /*** Interactive Functions                                                                  ***/
    /**********************************************************************************************/

    /**
     * @notice Registers or updates an integration with its facet address and selector wires.
     * @param  id     Unique identifier for the integration.
     * @param  config Facet address and array of selector wires to register.
     */
    function setIntegration(bytes32 id, Config calldata config) external;

    /**
     * @notice Removes an integration and all of its selector dispatches.
     * @param  id Unique identifier of the integration to remove.
     */
    function removeIntegration(bytes32 id) external;

    /**********************************************************************************************/
    /*** View/Pure Functions                                                                    ***/
    /**********************************************************************************************/

    /**
     * @notice Returns true if the contract supports the given interface.
     * @param  interfaceId The 4-byte interface identifier (ERC-165).
     * @return isSupported True if the interface is supported.
     */
    function supportsInterface(bytes4 interfaceId) external view returns (bool isSupported);

}


// ===== FILE: lib/diamond-pau/src/interfaces/IController.sol =====
// SPDX-License-Identifier: AGPL-3.0-or-later
pragma solidity ^0.8.34;

import { IEnumerableIntegrations } from "./IEnumerableIntegrations.sol";

/**
 * @title  IController
 * @notice Controller that manages integration lifecycle and routes incoming calls to facets via
 *         a dispatch table synced from the Beacon.
 */
interface IController is IEnumerableIntegrations {

    /**********************************************************************************************/
    /*** Custom Errors                                                                          ***/
    /**********************************************************************************************/

    /// @notice Thrown when a call selector is not wired to a facet.
    error CallSelectorNotWired(bytes4 callSelector);

    /// @notice Thrown when an argument array is empty.
    error EmptyArray();

    /// @notice Thrown when the call data length is less than 4.
    error InvalidCallDataLength(uint256 callDataLength);

    /// @notice Thrown when the caller is not an admin.
    error NotAdmin(address caller);

    /// @notice Thrown when the access controls is the zero address.
    error ZeroAccessControls();

    /// @notice Thrown when the beacon is the zero address.
    error ZeroBeacon();

    /// @notice Thrown when the proxy is the zero address.
    error ZeroProxy();

    /// @notice Thrown when the rate limits is the zero address.
    error ZeroRateLimits();

    /**********************************************************************************************/
    /*** Interactive Functions                                                                  ***/
    /**********************************************************************************************/

    /**
     * @notice Syncs integration configs from the Beacon and rebuilds the local dispatch table.
     * @param  ids Array of integration identifiers to update.
     */
    function updateIntegrations(bytes32[] calldata ids) external;

    /**
     * @notice Removes integrations and their selector dispatches from the controller.
     * @param  ids Array of integration identifiers to remove.
     */
    function removeIntegrations(bytes32[] calldata ids) external;

    /**********************************************************************************************/
    /*** Variables                                                                              ***/
    /**********************************************************************************************/

    /// @notice Address of the AccessControls contract.
    function accessControls() external view returns (address);

    /// @notice Address of the Beacon contract.
    function beacon() external view returns (address);

    /// @notice Address of the ALMProxy contract.
    function proxy() external view returns (address);

    /// @notice Address of the RateLimits contract.
    function rateLimits() external view returns (address);

}


// ===== FILE: lib/diamond-pau/src/ControllerSharedStorage.sol =====
// SPDX-License-Identifier: AGPL-3.0-or-later
pragma solidity ^0.8.34;

abstract contract ControllerSharedStorage {

    /**********************************************************************************************/
    /*** Shared Controller Storage Domain                                                       ***/
    /**********************************************************************************************/

    /// @custom:storage-location erc7201:sky.pau.storage.SharedController
    struct SharedControllerStorage {
        address accessControls;
        address proxy;
        address rateLimits;
    }

    // keccak256(abi.encode(uint256(keccak256("sky.pau.storage.SharedController")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 internal constant SHARED_CONTROLLER_STORAGE_LOCATION =
        0x77adf60bdbfedf206f8b8310f3d364080b7f61dcc0e46caac13c29bb1eb5cc00;

    function _getSharedControllerStorage()
        internal
        pure
        returns (SharedControllerStorage storage $)
    {
        assembly {
            $.slot := SHARED_CONTROLLER_STORAGE_LOCATION
        }
    }

}


// ===== FILE: lib/diamond-pau/lib/oz-upgradeable/contracts/proxy/utils/Initializable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (proxy/utils/Initializable.sol)

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
        // - construction: the contract is initialized at version 1 (no reininitialization) and the
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
     * @dev Returns a pointer to the storage namespace.
     */
    // solhint-disable-next-line var-name-mixedcase
    function _getInitializableStorage() private pure returns (InitializableStorage storage $) {
        assembly {
            $.slot := INITIALIZABLE_STORAGE
        }
    }
}


// ===== FILE: lib/diamond-pau/lib/openzeppelin-contracts/contracts/access/extensions/IAccessControlEnumerable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (access/extensions/IAccessControlEnumerable.sol)

pragma solidity ^0.8.20;

import {IAccessControl} from "../IAccessControl.sol";

/**
 * @dev External interface of AccessControlEnumerable declared to support ERC165 detection.
 */
interface IAccessControlEnumerable is IAccessControl {
    /**
     * @dev Returns one of the accounts that have `role`. `index` must be a
     * value between 0 and {getRoleMemberCount}, non-inclusive.
     *
     * Role bearers are not sorted in any particular way, and their ordering may
     * change at any point.
     *
     * WARNING: When using {getRoleMember} and {getRoleMemberCount}, make sure
     * you perform all queries on the same block. See the following
     * https://forum.openzeppelin.com/t/iterating-over-elements-on-enumerableset-in-openzeppelin-contracts/2296[forum post]
     * for more information.
     */
    function getRoleMember(bytes32 role, uint256 index) external view returns (address);

    /**
     * @dev Returns the number of accounts that have `role`. Can be used
     * together with {getRoleMember} to enumerate all bearers of a role.
     */
    function getRoleMemberCount(bytes32 role) external view returns (uint256);
}


// ===== FILE: lib/diamond-pau/src/interfaces/IEnumerableIntegrations.sol =====
// SPDX-License-Identifier: AGPL-3.0-or-later
pragma solidity ^0.8.34;

/**
 * @title  IEnumerableIntegrations
 * @notice Shared interface for enumerating integrations, configs, and dispatch routing used by
 *         both the Beacon and the Controller.
 */
interface IEnumerableIntegrations {

    /**********************************************************************************************/
    /*** Structs                                                                                ***/
    /**********************************************************************************************/

    /**
     * @notice Configuration for an integration: the facet contract and its selector wires.
     * @param  facet Address of the facet contract.
     * @param  wires Array of selector mappings from call selectors to delegate selectors.
     */
    struct Config {
        address facet;
        Wire[]  wires;
    }

    /**
     * @notice Resolved dispatch entry mapping a call selector to its target facet and delegate
     *         selector.
     * @param  facet            Address of the facet contract to delegate to.
     * @param  delegateSelector The 4-byte selector to use in the delegatecall.
     */
    struct Dispatch {
        address facet;
        bytes4  delegateSelector;
    }

    /**
     * @notice Full integration record containing its identifier and configuration.
     * @param  id     Unique identifier for the integration.
     * @param  config The integration's facet and wiring configuration.
     */
    struct Integration {
        bytes32 id;
        Config  config;
    }

    /**
     * @notice Selector wire mapping an external call selector to an internal delegate selector
     *         on the facet.
     * @param  callSelector     The 4-byte selector used in external calls to the controller.
     * @param  delegateSelector The 4-byte selector forwarded to the facet via delegatecall.
     */
    struct Wire {
        bytes4 callSelector;
        bytes4 delegateSelector;
    }

    /**********************************************************************************************/
    /*** Events                                                                                 ***/
    /**********************************************************************************************/

    /**
     * @notice Emitted when an integration is registered or updated.
     * @param  id     Unique identifier for the integration.
     * @param  config The facet address and selector wires that were set.
     */
    event IntegrationSet(bytes32 indexed id, Config config);

    /**
     * @notice Emitted when an integration is removed.
     * @param  id Unique identifier of the removed integration.
     */
    event IntegrationRemoved(bytes32 indexed id);

    /**********************************************************************************************/
    /*** Custom Errors                                                                          ***/
    /**********************************************************************************************/

    /// @notice Thrown when a call selector is already wired to a facet.
    error CallSelectorAlreadyWired(bytes4 callSelector);

    /// @notice Thrown when the facet has no code.
    error EmptyFacet();

    /// @notice Thrown when the integration is not found.
    error IntegrationNotFound(bytes32 id);

    /**********************************************************************************************/
    /*** Variables                                                                              ***/
    /**********************************************************************************************/

    /// @notice Returns all registered integrations with their configs.
    function integrations() external view returns (Integration[] memory);

    /**********************************************************************************************/
    /*** View/Pure Functions                                                                    ***/
    /**********************************************************************************************/

    /**
     * @notice Returns the config for a single integration.
     * @param  integrationId Unique identifier of the integration.
     * @return config        The facet address and selector wires for the integration.
     */
    function getConfig(bytes32 integrationId) external view returns (Config memory config);

    /**
     * @notice Returns configs for multiple integrations in a single call.
     * @param  integrationIds Array of integration identifiers.
     * @return configs        Array of configs corresponding to the given identifiers.
     */
    function getConfigs(bytes32[] calldata integrationIds)
        external
        view
        returns (Config[] memory configs);

    /**
     * @notice Returns the dispatch entry for a call selector.
     * @param  callSelector The 4-byte external call selector.
     * @return dispatch     The facet address and delegate selector for the given call selector.
     */
    function getDispatch(bytes4 callSelector) external view returns (Dispatch memory dispatch);

    /**
     * @notice Returns dispatch entries for multiple call selectors in a single call.
     * @param  callSelectors Array of 4-byte external call selectors.
     * @return dispatches    Array of dispatch entries corresponding to the given selectors.
     */
    function getDispatches(bytes4[] calldata callSelectors)
        external
        view
        returns (Dispatch[] memory dispatches);

}


// ===== FILE: lib/diamond-pau/lib/openzeppelin-contracts/contracts/access/IAccessControl.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (access/IAccessControl.sol)

pragma solidity ^0.8.20;

/**
 * @dev External interface of AccessControl declared to support ERC165 detection.
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
     * {RoleAdminChanged} not being emitted signaling this.
     */
    event RoleAdminChanged(bytes32 indexed role, bytes32 indexed previousAdminRole, bytes32 indexed newAdminRole);

    /**
     * @dev Emitted when `account` is granted `role`.
     *
     * `sender` is the account that originated the contract call, an admin role
     * bearer except when using {AccessControl-_setupRole}.
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
