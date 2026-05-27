// ===== FILE: _axelar-network/axelar-gmp-sdk-solidity/contracts/interfaces/IProxy.sol =====
// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

// General interface for upgradable contracts
interface IProxy {
    error InvalidOwner();
    error InvalidImplementation();
    error SetupFailed();
    error NotOwner();
    error AlreadyInitialized();

    function implementation() external view returns (address);

    function setup(bytes calldata setupParams) external;
}


// ===== FILE: _axelar-network/axelar-gmp-sdk-solidity/contracts/upgradable/BaseProxy.sol =====
// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

import { IProxy } from '../interfaces/IProxy.sol';

/**
 * @title BaseProxy Contract
 * @dev This abstract contract implements a basic proxy that stores an implementation address. Fallback function
 * calls are delegated to the implementation. This contract is meant to be inherited by other proxy contracts.
 */
abstract contract BaseProxy is IProxy {
    // bytes32(uint256(keccak256('eip1967.proxy.implementation')) - 1)
    bytes32 internal constant _IMPLEMENTATION_SLOT = 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc;
    // keccak256('owner')
    bytes32 internal constant _OWNER_SLOT = 0x02016836a56b71f0d02689e69e326f4f4c1b9057164ef592671cf0d37c8040c0;

    /**
     * @dev Returns the current implementation address.
     * @return implementation_ The address of the current implementation contract
     */
    function implementation() public view virtual returns (address implementation_) {
        assembly {
            implementation_ := sload(_IMPLEMENTATION_SLOT)
        }
    }

    /**
     * @dev Shadows the setup function of the implementation contract so it can't be called directly via the proxy.
     * @param params The setup parameters for the implementation contract.
     */
    function setup(bytes calldata params) external {}

    /**
     * @dev Returns the contract ID. It can be used as a check during upgrades. Meant to be implemented in derived contracts.
     * @return bytes32 The contract ID
     */
    function contractId() internal pure virtual returns (bytes32);

    /**
     * @dev Fallback function. Delegates the call to the current implementation contract.
     */
    fallback() external payable virtual {
        address implementation_ = implementation();
        assembly {
            calldatacopy(0, 0, calldatasize())

            let result := delegatecall(gas(), implementation_, 0, calldatasize(), 0, 0)
            returndatacopy(0, 0, returndatasize())

            switch result
            case 0 {
                revert(0, returndatasize())
            }
            default {
                return(0, returndatasize())
            }
        }
    }

    /**
     * @dev Payable fallback function. Can be overridden in derived contracts.
     */
    receive() external payable virtual {}
}


// ===== FILE: contracts/interfaces/IBaseTokenManager.sol =====
// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

/**
 * @title IBaseTokenManager
 * @notice This contract is defines the base token manager interface implemented by all token managers.
 */
interface IBaseTokenManager {
    /**
     * @notice A function that returns the token id.
     */
    function interchainTokenId() external view returns (bytes32);

    /**
     * @notice A function that should return the address of the token.
     * Must be overridden in the inheriting contract.
     * @return address address of the token.
     */
    function tokenAddress() external view returns (address);

    /**
     * @notice A function that should return the token address from the init params.
     */
    function getTokenAddressFromParams(bytes calldata params) external pure returns (address);
}


// ===== FILE: contracts/interfaces/ITokenManagerImplementation.sol =====
// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

/**
 * @title ITokenManagerImplementation Interface
 * @notice Interface for returning the token manager implementation type.
 */
interface ITokenManagerImplementation {
    /**
     * @notice Returns the implementation address for a given token manager type.
     * @param tokenManagerType The type of token manager.
     * @return tokenManagerAddress_ The address of the token manager implementation.
     */
    function tokenManagerImplementation(uint256 tokenManagerType) external view returns (address tokenManagerAddress_);
}


// ===== FILE: contracts/interfaces/ITokenManagerProxy.sol =====
// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

import { IProxy } from '@axelar-network/axelar-gmp-sdk-solidity/contracts/interfaces/IProxy.sol';

/**
 * @title ITokenManagerProxy Interface
 * @notice This interface is for a proxy for token manager contracts.
 */
interface ITokenManagerProxy is IProxy {
    error ZeroAddress();

    /**
     * @notice Returns implementation type of this token manager.
     * @return uint256 The implementation type of this token manager.
     */
    function implementationType() external view returns (uint256);

    /**
     * @notice Returns the interchain token ID of the token manager.
     * @return bytes32 The interchain token ID of the token manager.
     */
    function interchainTokenId() external view returns (bytes32);

    /**
     * @notice Returns token address that this token manager manages.
     * @return address The token address.
     */
    function tokenAddress() external view returns (address);

    /**
     * @notice Returns implementation type and token address.
     * @return uint256 The implementation type.
     * @return address The token address.
     */
    function getImplementationTypeAndTokenAddress() external view returns (uint256, address);
}


// ===== FILE: contracts/proxies/TokenManagerProxy.sol =====
// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

import { IProxy } from '@axelar-network/axelar-gmp-sdk-solidity/contracts/interfaces/IProxy.sol';
import { BaseProxy } from '@axelar-network/axelar-gmp-sdk-solidity/contracts/upgradable/BaseProxy.sol';

import { IBaseTokenManager } from '../interfaces/IBaseTokenManager.sol';
import { ITokenManagerProxy } from '../interfaces/ITokenManagerProxy.sol';
import { ITokenManagerImplementation } from '../interfaces/ITokenManagerImplementation.sol';

/**
 * @title TokenManagerProxy
 * @notice This contract is a proxy for token manager contracts.
 * @dev This contract implements BaseProxy and ITokenManagerProxy.
 */
contract TokenManagerProxy is BaseProxy, ITokenManagerProxy {
    bytes32 private constant CONTRACT_ID = keccak256('token-manager');

    address public immutable interchainTokenService;
    uint256 public immutable implementationType;
    bytes32 public immutable interchainTokenId;
    address public immutable tokenAddress;

    /**
     * @notice Constructs the TokenManagerProxy contract.
     * @param interchainTokenService_ The address of the interchain token service.
     * @param implementationType_ The token manager type.
     * @param tokenId The identifier for the token.
     * @param params The initialization parameters for the token manager contract.
     */
    constructor(address interchainTokenService_, uint256 implementationType_, bytes32 tokenId, bytes memory params) {
        if (interchainTokenService_ == address(0)) revert ZeroAddress();

        interchainTokenService = interchainTokenService_;
        implementationType = implementationType_;
        interchainTokenId = tokenId;

        address implementation_ = _tokenManagerImplementation(interchainTokenService_, implementationType_);
        if (implementation_ == address(0)) revert InvalidImplementation();

        (bool success, ) = implementation_.delegatecall(abi.encodeWithSelector(IProxy.setup.selector, params));
        if (!success) revert SetupFailed();

        tokenAddress = IBaseTokenManager(implementation_).getTokenAddressFromParams(params);
    }

    /**
     * @notice Getter for the contract id.
     * @return bytes32 The contract id.
     */
    function contractId() internal pure override returns (bytes32) {
        return CONTRACT_ID;
    }

    /**
     * @notice Returns implementation type and token address.
     * @return implementationType_ The implementation type.
     * @return tokenAddress_ The token address.
     */
    function getImplementationTypeAndTokenAddress() external view returns (uint256 implementationType_, address tokenAddress_) {
        implementationType_ = implementationType;
        tokenAddress_ = tokenAddress;
    }

    /**
     * @notice Returns the address of the current implementation.
     * @return implementation_ The address of the current implementation.
     */
    function implementation() public view override(BaseProxy, IProxy) returns (address implementation_) {
        implementation_ = _tokenManagerImplementation(interchainTokenService, implementationType);
    }

    /**
     * @notice Returns the implementation address from the interchain token service for the provided type.
     * @param interchainTokenService_ The address of the interchain token service.
     * @param implementationType_ The token manager type.
     * @return implementation_ The address of the implementation.
     */
    function _tokenManagerImplementation(
        address interchainTokenService_,
        uint256 implementationType_
    ) internal view returns (address implementation_) {
        implementation_ = ITokenManagerImplementation(interchainTokenService_).tokenManagerImplementation(implementationType_);
    }
}
