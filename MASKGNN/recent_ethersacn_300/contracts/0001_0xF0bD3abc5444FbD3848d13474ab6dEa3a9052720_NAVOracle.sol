// ===== FILE: src/modules/NAVOracle.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { ECDSA } from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import { IERC20 } from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import { IERC20Metadata } from "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import { CoreAccess } from "../base/CoreAccess.sol";
import { INAVOracle } from "../interfaces/INAVOracle.sol";
import { PriceEntry, SignedPriceBundle } from "../interfaces/Types.sol";
import { FixedPoint } from "../libraries/FixedPoint.sol";
import { Roles } from "../libraries/Roles.sol";

interface ICoreVaultAllocation {
    function getTargetAllocation() external view returns (address[] memory tokens, uint256[] memory weights);
}

/// @title NAVOracle
/// @dev  push-on-touch model
contract NAVOracle is CoreAccess, INAVOracle {
    using ECDSA for bytes32;

    bytes32 public constant PRICE_ENTRY_TYPEHASH =
        keccak256("PriceEntry(address asset,uint256 price,uint8 priceDecimals,uint64 asOfTimestamp)");
    bytes32 public constant BUNDLE_TYPEHASH = keccak256(
        "SignedPriceBundle(uint64 round,uint64 deadline,PriceEntry[] entries,bytes32 assetSetDigest)PriceEntry(address asset,uint256 price,uint8 priceDecimals,uint64 asOfTimestamp)"
    );

    //EIP712_DOMAIN_TYPEHASH =
    //                    keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)")
    //        = 0x8b73c3c69bb8fe3d512ecc4cf759cc79239f7b179b0ffacaa9a75d522b39400f
    //NAV_ORACLE_NAME_HASH =
    //                    keccak256("CoreVault-NAVOracle")
    //        = 0xaea4d67c2e12725f70eba049f95c5c59c259832263d438ed502ac220001bbc42

    //NAV_ORACLE_VERSION_HASH =
    //                    keccak256("1")
    //        = 0xc89efdaa54c0f20c7adf612882df0950f5a951637e0307cdcb4c672f298b8bc6

    bytes32 public constant EIP712_DOMAIN_TYPEHASH =
            0x8b73c3c69bb8fe3d512ecc4cf759cc79239f7b179b0ffacaa9a75d522b39400f;
    bytes32 public constant NAV_ORACLE_NAME_HASH =
        0xaea4d67c2e12725f70eba049f95c5c59c259832263d438ed502ac220001bbc42;
    bytes32 public constant NAV_ORACLE_VERSION_HASH =
        0xc89efdaa54c0f20c7adf612882df0950f5a951637e0307cdcb4c672f298b8bc6;

    uint256 public maxStaleness;
    uint256 public maxRoundJump;
    address public coreVault;

    uint64 private _latestRound;
    mapping(uint256 => bytes32) private _bundleHashByRound;
    mapping(address => uint256) private _priceByAsset;
    mapping(address => bool) public trustedSigner;
    mapping(address => bool) public authorizedCaller;
    bool private _navOracleInitialized;

    event PriceBundleIngested(uint64 indexed round, address indexed signer);
    event TrustedSignerSet(address indexed signer, bool trusted);
    event AuthorizedCallerSet(address indexed caller, bool authorized);
    event CoreVaultSet(address indexed coreVault);
    event MaxRoundJumpSet(uint256 maxRoundJump);

    constructor(address admin, uint256 maxStaleness_, uint256 maxRoundJump_)
        CoreAccess(admin)
    {
        if (admin != address(0)) _initializeNavOracle(maxStaleness_, maxRoundJump_);
    }

    function initialize(address admin, uint256 maxStaleness_, uint256 maxRoundJump_) external {
        _initializeCoreAccess(admin);
        _initializeNavOracle(maxStaleness_, maxRoundJump_);
    }

    function _initializeNavOracle(uint256 maxStaleness_, uint256 maxRoundJump_) internal {
        require(!_navOracleInitialized, "NAVOracle: initialized");
        require(maxStaleness_ > 0 && maxRoundJump_ > 0, "NAVOracle: bad params");
        _navOracleInitialized = true;
        maxStaleness = maxStaleness_;
        maxRoundJump = maxRoundJump_;
    }

    function setTrustedSigner(address signer, bool trusted) external onlyAdmin {
        trustedSigner[signer] = trusted;
        emit TrustedSignerSet(signer, trusted);
    }

    function setAuthorizedCaller(address caller, bool authorized) external onlyAdmin {
        authorizedCaller[caller] = authorized;
        emit AuthorizedCallerSet(caller, authorized);
    }

    function setCoreVault(address coreVault_) external onlyAdmin {
        require(coreVault_ != address(0), "NAVOracle: zero core vault");
        coreVault = coreVault_;
        emit CoreVaultSet(coreVault_);
    }

    function setMaxRoundJump(uint256 maxRoundJump_) external onlyAdmin {
        require(maxRoundJump_ > 0, "NAVOracle: bad params");
        maxRoundJump = maxRoundJump_;
        emit MaxRoundJumpSet(maxRoundJump_);
    }

    function ingest(SignedPriceBundle calldata bundle) external returns (uint64 round) {
        require(
            (authorizedCaller[msg.sender] && hasRole(Roles.NAV_CALLER, msg.sender))
                || hasRole(Roles.OPERATOR, msg.sender),
            "NAVOracle: unauthorized caller"
        );
        require(block.timestamp <= bundle.deadline, "NAVOracle: expired");
        (address[] memory tokens, uint256[] memory weights) = _targetAllocation(msg.sender);
        _validateAssetSet(bundle, tokens, weights);
        bytes32 bundleHash = hashBundle(bundle);
        address recovered = bundleHash.recover(bundle.v, bundle.r, bundle.s);
        require(recovered == bundle.signer && trustedSigner[recovered], "NAVOracle: bad signer");

        if (bundle.round == _latestRound && _latestRound != 0) {
            require(_bundleHashByRound[bundle.round] == bundleHash, "NAVOracle: round changed");
            return bundle.round;
        }

        require(bundle.round > _latestRound, "NAVOracle: old round");
        if (_latestRound != 0) {
            require(maxRoundJump <= type(uint64).max, "NAVOracle: round jump overflow");
            // casting to 'uint64' is safe because maxRoundJump is checked above.
            // forge-lint: disable-next-line(unsafe-typecast)
            require(bundle.round <= _latestRound + uint64(maxRoundJump), "NAVOracle: round jump");
        }
        _validateBundleTimestamps(bundle);
        for (uint256 i; i < bundle.entries.length; ++i) {
            PriceEntry calldata entry = bundle.entries[i];
            _priceByAsset[entry.asset] = _normalizePrice(entry.price, entry.priceDecimals);
        }

        _latestRound = bundle.round;
        _bundleHashByRound[bundle.round] = bundleHash;
        emit PriceBundleIngested(bundle.round, recovered);
        return bundle.round;
    }

    function latestRound() external view returns (uint64) {
        return _latestRound;
    }

    function priceOf(address asset_) external view returns (uint256) {
        return _priceByAsset[asset_];
    }

    /// @notice EIP-712
    function domainSeparator() public view returns (bytes32) {
        return _domainSeparatorHash(block.chainid, address(this));
    }

    function hashBundle(SignedPriceBundle calldata bundle) public view returns (bytes32) {
        bytes32[] memory entryHashes = new bytes32[](bundle.entries.length);
        for (uint256 i; i < bundle.entries.length; ++i) {
            PriceEntry calldata entry = bundle.entries[i];
            // PriceEntry(address asset,uint256 price,uint8 priceDecimals,uint64 asOfTimestamp) 的 hash。
            // forge-lint: disable-next-line(asm-keccak256)
            entryHashes[i] = keccak256(
                abi.encode(PRICE_ENTRY_TYPEHASH, entry.asset, entry.price, entry.priceDecimals, entry.asOfTimestamp)
            );
        }
        bytes32 entriesHash = _hashEntryHashes(entryHashes);
        bytes32 structHash =
            _bundleStructHash(BUNDLE_TYPEHASH, bundle.round, bundle.deadline, entriesHash, bundle.assetSetDigest);
        return ECDSA.toTypedDataHash(domainSeparator(), structHash);
    }

    function recoverBundleSigner(SignedPriceBundle calldata bundle)
        external
        view
        returns (bytes32 digest, address recovered, bool signerMatches, bool recoveredTrusted)
    {
        digest = hashBundle(bundle);
        recovered = digest.recover(bundle.v, bundle.r, bundle.s);
        signerMatches = recovered == bundle.signer;
        recoveredTrusted = trustedSigner[recovered];
    }

    function recomputeTotalAssetsFromBundle(
        address vault,
        address usdc,
        SignedPriceBundle calldata bundle
    ) external view returns (uint256 totalAssetsValue) {
        require(block.timestamp <= bundle.deadline, "NAVOracle: expired");
        (address[] memory tokens, uint256[] memory weights) = _targetAllocation(vault);
        _validateAssetSet(bundle, tokens, weights);
        bytes32 bundleHash = hashBundle(bundle);
        address recovered = bundleHash.recover(bundle.v, bundle.r, bundle.s);
        require(recovered == bundle.signer && trustedSigner[recovered], "NAVOracle: bad signer");
        _validateBundleTimestamps(bundle);

        totalAssetsValue = FixedPoint.usdcToWad(IERC20(usdc).balanceOf(vault));
        for (uint256 i; i < tokens.length; ++i) {
            address token = tokens[i];
            if (token == usdc) continue;
            uint256 balance = IERC20(token).balanceOf(vault);
            if (balance == 0) continue;
            uint256 price = _priceInBundle(bundle, token);
            require(price > 0, "NAVOracle: missing price");
            totalAssetsValue += balance * price / _assetUnit(token);
        }
    }

    function _hashEntryHashes(bytes32[] memory entryHashes) internal pure returns (bytes32 result) {
        assembly ("memory-safe") {
            result := keccak256(add(entryHashes, 0x20), mul(mload(entryHashes), 0x20))
        }
    }

    function _domainSeparatorHash(uint256 chainId, address verifyingContract) internal pure returns (bytes32 result) {
        assembly ("memory-safe") {
            let ptr := mload(0x40)
            mstore(ptr, EIP712_DOMAIN_TYPEHASH)
            mstore(add(ptr, 0x20), NAV_ORACLE_NAME_HASH)
            mstore(add(ptr, 0x40), NAV_ORACLE_VERSION_HASH)
            mstore(add(ptr, 0x60), chainId)
            mstore(add(ptr, 0x80), and(verifyingContract, 0xffffffffffffffffffffffffffffffffffffffff))
            result := keccak256(ptr, 0xa0)
        }
    }

    function _bundleStructHash(
        bytes32 typehash,
        uint64 round,
        uint64 deadline,
        bytes32 entriesHash,
        bytes32 digest
    ) internal pure returns (bytes32 result) {
        assembly ("memory-safe") {
            let ptr := mload(0x40)
            mstore(ptr, typehash)
            mstore(add(ptr, 0x20), round)
            mstore(add(ptr, 0x40), deadline)
            mstore(add(ptr, 0x60), entriesHash)
            mstore(add(ptr, 0x80), digest)
            result := keccak256(ptr, 0xa0)
        }
    }

    function _normalizePrice(uint256 price, uint8 priceDecimals) internal pure returns (uint256 normalized) {
        require(price > 0, "NAVOracle: zero price");
        require(priceDecimals <= 36, "NAVOracle: bad price decimals");
        if (priceDecimals == 18) normalized = price;
        else if (priceDecimals < 18) normalized = price * (10 ** uint256(18 - priceDecimals));
        else normalized = price / (10 ** uint256(priceDecimals - 18));
        require(normalized > 0, "NAVOracle: zero normalized price");
    }

    function _validatePriceTimestamp(uint64 asOfTimestamp) internal view {
        require(asOfTimestamp <= block.timestamp, "NAVOracle: future price");
        require(block.timestamp <= uint256(asOfTimestamp) + maxStaleness, "NAVOracle: stale price");
    }

    function _validateBundleTimestamps(SignedPriceBundle calldata bundle) internal view {
        for (uint256 i; i < bundle.entries.length; ++i) {
            _validatePriceTimestamp(bundle.entries[i].asOfTimestamp);
        }
    }

    function _targetAllocation(address sourceHint)
        internal
        view
        returns (address[] memory tokens, uint256[] memory weights)
    {
        address source = coreVault == address(0) ? sourceHint : coreVault;
        require(source != address(0), "NAVOracle: core vault unset");
        return ICoreVaultAllocation(source).getTargetAllocation();
    }

    function _validateAssetSet(
        SignedPriceBundle calldata bundle,
        address[] memory tokens,
        uint256[] memory weights
    ) internal pure {
        require(tokens.length != 0 && bundle.entries.length == tokens.length, "NAVOracle: bad asset entries");
        require(bundle.assetSetDigest == keccak256(abi.encode(tokens, weights)), "NAVOracle: bad asset set");
        for (uint256 i; i < tokens.length; ++i) {
            require(bundle.entries[i].asset == tokens[i], "NAVOracle: bad asset entries");
        }
    }

    function _priceInBundle(SignedPriceBundle calldata bundle, address asset_) internal pure returns (uint256) {
        for (uint256 i; i < bundle.entries.length; ++i) {
            PriceEntry calldata entry = bundle.entries[i];
            if (entry.asset == asset_) return _normalizePrice(entry.price, entry.priceDecimals);
        }
        return 0;
    }

    function _assetUnit(address token) internal view returns (uint256) {
        uint8 decimals = IERC20Metadata(token).decimals();
        require(decimals <= 36, "NAVOracle: bad decimals");
        return 10 ** uint256(decimals);
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/cryptography/ECDSA.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

library ECDSA {
    function recover(bytes32 hash, uint8 v, bytes32 r, bytes32 s) internal pure returns (address) {
        address signer = ecrecover(hash, v, r, s);
        require(signer != address(0), "ECDSA: invalid signature");
        return signer;
    }

    function toTypedDataHash(bytes32 domainSeparator, bytes32 structHash) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked("\x19\x01", domainSeparator, structHash));
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/token/ERC20/IERC20.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 value) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}


// ===== FILE: lib/openzeppelin-contracts/contracts/token/ERC20/extensions/IERC20Metadata.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "../IERC20.sol";

interface IERC20Metadata is IERC20 {
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
}


// ===== FILE: src/base/CoreAccess.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { IVaultAuthority } from "../interfaces/IVaultAuthority.sol";
import { Roles } from "../libraries/Roles.sol";
import { UUPSUpgradeable } from "oz-v5/proxy/utils/UUPSUpgradeable.sol";

/// @title CoreAccess
abstract contract CoreAccess is UUPSUpgradeable {
    IVaultAuthority public authority;
    bool private _coreAccessInitialized;

    event AuthoritySet(address indexed authority);

    constructor(address authority_) {
        if (authority_ == address(0)) {
            _coreAccessInitialized = true;
        } else {
            _initializeCoreAccess(authority_);
        }
    }

    function _initializeCoreAccess(address authority_) internal {
        require(!_coreAccessInitialized, "CoreAccess: initialized");
        require(authority_ != address(0), "CoreAccess: zero authority");
        _coreAccessInitialized = true;
        authority = IVaultAuthority(authority_);
        emit AuthoritySet(authority_);
    }

    modifier onlyAdmin() {
        _checkRole(Roles.DEFAULT_ADMIN_ROLE, msg.sender);
        _;
    }

    modifier onlySuperAdmin() {
        _checkRole(Roles.SUPER_ADMIN_ROLE, msg.sender);
        _;
    }

    modifier onlyRole(bytes32 role) {
        _checkRole(role, msg.sender);
        _;
    }

    function _authorizeUpgrade(address) internal override onlySuperAdmin { }

    function hasRole(bytes32 role, address account) public view returns (bool) {
        return address(authority) != address(0) && authority.hasRole(role, account);
    }

    function _checkRole(bytes32 role, address account) internal view {
        require(hasRole(role, account), "CoreAccess: missing role");
    }
}


// ===== FILE: src/interfaces/INAVOracle.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {SignedPriceBundle} from "./Types.sol";

interface INAVOracle {
    function ingest(SignedPriceBundle calldata bundle) external returns (uint64 round);
    function latestRound() external view returns (uint64);
    function priceOf(address asset) external view returns (uint256);
    function domainSeparator() external view returns (bytes32);
    function hashBundle(SignedPriceBundle calldata bundle) external view returns (bytes32);
    function recoverBundleSigner(SignedPriceBundle calldata bundle)
        external
        view
        returns (bytes32 digest, address recovered, bool signerMatches, bool recoveredTrusted);
    function recomputeTotalAssetsFromBundle(
        address vault,
        address usdc,
        SignedPriceBundle calldata bundle
    ) external view returns (uint256 totalAssetsValue);
}


// ===== FILE: src/interfaces/Types.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

struct PriceEntry {
    address asset;
    uint256 price;
    uint8 priceDecimals;
    uint64 asOfTimestamp;
}

struct SignedPriceBundle {
    uint64 round;
    uint64 deadline;
    PriceEntry[] entries;
    bytes32 assetSetDigest;
    address signer;
    uint8 v;
    bytes32 r;
    bytes32 s;
}

struct OndoTarget {
    address token;
    uint16 bps;
}

struct BatchHint {
    uint16 maxSlippageBps;
    uint16 maxLossPercentBps;
}

struct Trade {
    address tokenIn;
    address tokenOut;
    uint256 amountIn;
    uint256 minAmountOut;
}


struct RebalanceBuy {
    address tokenIn;
    address tokenOut;
    uint256 amountIn;
    uint256 minAmountOut;
}

enum RequestStatus {
    Executing,
    Completed,
    Failed
}

enum BatchStatus {
    PendingProxy,
    Settled,
    Failed
}

enum RebalanceTrigger {
    Active,
    PassivePeriodic,
    PassiveRiskEvent
}

struct PassiveRebalanceParams {
    Trade[] sells;
    RebalanceBuy[] buys;
    address[] targetTokens;
    uint256[] targetWeights;
    RebalanceTrigger trigger;
    uint8 riskLevel;
    bytes32 reasonHash;
    SignedPriceBundle priceBundle;
}


// ===== FILE: src/libraries/FixedPoint.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { Math } from "@openzeppelin/contracts/utils/math/Math.sol";

/// @title FixedPoint
library FixedPoint {
    uint256 internal constant WAD = 1e18;
    uint256 internal constant BPS = 10_000;
    uint256 internal constant USDC_SCALE = 1e12;

    function usdcToWad(uint256 amount) internal pure returns (uint256) {
        return amount * USDC_SCALE;
    }

    function wadToUsdc(uint256 amount) internal pure returns (uint256) {
        return amount / USDC_SCALE;
    }

    function mulDiv(uint256 x, uint256 y, uint256 denominator) internal pure returns (uint256) {
        return Math.mulDiv(x, y, denominator);
    }

    function bpsOf(uint256 amount, uint256 bps) internal pure returns (uint256) {
        return amount * bps / BPS;
    }
}


// ===== FILE: src/libraries/Roles.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title Roles
library Roles {
    bytes32 internal constant DEFAULT_ADMIN_ROLE = 0x00;
    // 0x7613a25ecc738585a232ad50a301178f12b3ba8887d13e138b523c4269c47689
    bytes32 internal constant SUPER_ADMIN_ROLE = keccak256("SUPER_ADMIN_ROLE");
    // 0x139c2898040ef16910dc9f44dc697df79363da767d8bc92f2e310312b816e46d
    bytes32 internal constant PAUSE_ROLE = keccak256("PAUSE_ROLE");
    // 0x0aec9b08bc8c2cb62a91f52d33e3d77da4b3f3a63fc8b542a93abe3902ba929c
    bytes32 internal constant CURATOR = keccak256("CURATOR");
    // 0x390ae1081063aad083f9b4572bbfcc286af7814a39b668db2927a1186deec81b
    bytes32 internal constant MONITOR = keccak256("MONITOR");
    // 0x523a704056dcd17bcf83bed8b68c59416dac1119be77755efe3bde0a64e46e0c
    bytes32 internal constant OPERATOR = keccak256("OPERATOR");
    // 0x8a639721758084b2ff869ed05cee658aab46e6bdc69bca1b006bae9f55613c59
    bytes32 internal constant FEE_CONTROLLER = keccak256("FEE_CONTROLLER");
    // 0x262c70cb68844873654dc54487b634cb00850c1e13c785cd0d96a2b89b829472
    bytes32 internal constant TOKEN_MINTER = keccak256("TOKEN_MINTER");
    // 0x47bc4cfe7b2bde7da37514f11daac8409232278f94544d2e3175143878762e9f
    bytes32 internal constant NAV_CALLER = keccak256("NAV_CALLER");
    //
    bytes32 internal constant ONDO_TARGET_MANAGER = keccak256("ONDO_TARGET_MANAGER");
}


// ===== FILE: src/interfaces/IVaultAuthority.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IVaultAuthority {
    event RoleGranted(bytes32 indexed role, address indexed account, address indexed sender);
    event RoleRevoked(bytes32 indexed role, address indexed account, address indexed sender);

    function hasRole(bytes32 role, address account) external view returns (bool);
    function grantAdmin(address account) external;
    function revokeAdmin(address account) external;
    function grantRole(bytes32 role, address account) external;
    function revokeRole(bytes32 role, address account) external;
    function renounceRole(bytes32 role, address account) external;
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/proxy/utils/UUPSUpgradeable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.5.0) (proxy/utils/UUPSUpgradeable.sol)

pragma solidity ^0.8.22;

import {IERC1822Proxiable} from "../../interfaces/draft-IERC1822.sol";
import {ERC1967Utils} from "../ERC1967/ERC1967Utils.sol";

/**
 * @dev An upgradeability mechanism designed for UUPS proxies. The functions included here can perform an upgrade of an
 * {ERC1967Proxy}, when this contract is set as the implementation behind such a proxy.
 *
 * A security mechanism ensures that an upgrade does not turn off upgradeability accidentally, although this risk is
 * reinstated if the upgrade retains upgradeability but removes the security mechanism, e.g. by replacing
 * `UUPSUpgradeable` with a custom implementation of upgrades.
 *
 * The {_authorizeUpgrade} function must be overridden to include access restriction to the upgrade mechanism.
 *
 * @custom:stateless
 */
abstract contract UUPSUpgradeable is IERC1822Proxiable {
    /// @custom:oz-upgrades-unsafe-allow state-variable-immutable
    address private immutable __self = address(this);

    /**
     * @dev The version of the upgrade interface of the contract. If this getter is missing, both `upgradeTo(address)`
     * and `upgradeToAndCall(address,bytes)` are present, and `upgradeTo` must be used if no function should be called,
     * while `upgradeToAndCall` will invoke the `receive` function if the second argument is the empty byte string.
     * If the getter returns `"5.0.0"`, only `upgradeToAndCall(address,bytes)` is present, and the second argument must
     * be the empty byte string if no function should be called, making it impossible to invoke the `receive` function
     * during an upgrade.
     */
    string public constant UPGRADE_INTERFACE_VERSION = "5.0.0";

    /**
     * @dev The call is from an unauthorized context.
     */
    error UUPSUnauthorizedCallContext();

    /**
     * @dev The storage `slot` is unsupported as a UUID.
     */
    error UUPSUnsupportedProxiableUUID(bytes32 slot);

    /**
     * @dev Check that the execution is being performed through a delegatecall call and that the execution context is
     * a proxy contract with an implementation (as defined in ERC-1967) pointing to self. This should only be the case
     * for UUPS and transparent proxies that are using the current contract as their implementation. Execution of a
     * function through ERC-1167 minimal proxies (clones) would not normally pass this test, but is not guaranteed to
     * fail.
     */
    modifier onlyProxy() {
        _checkProxy();
        _;
    }

    /**
     * @dev Check that the execution is not being performed through a delegate call. This allows a function to be
     * callable on the implementing contract but not through proxies.
     */
    modifier notDelegated() {
        _checkNotDelegated();
        _;
    }

    /**
     * @dev Implementation of the ERC-1822 {proxiableUUID} function. This returns the storage slot used by the
     * implementation. It is used to validate the implementation's compatibility when performing an upgrade.
     *
     * IMPORTANT: A proxy pointing at a proxiable contract should not be considered proxiable itself, because this risks
     * bricking a proxy that upgrades to it, by delegating to itself until out of gas. Thus it is critical that this
     * function revert if invoked through a proxy. This is guaranteed by the `notDelegated` modifier.
     */
    function proxiableUUID() external view notDelegated returns (bytes32) {
        return ERC1967Utils.IMPLEMENTATION_SLOT;
    }

    /**
     * @dev Upgrade the implementation of the proxy to `newImplementation`, and subsequently execute the function call
     * encoded in `data`.
     *
     * Calls {_authorizeUpgrade}.
     *
     * Emits an {Upgraded} event.
     *
     * @custom:oz-upgrades-unsafe-allow-reachable delegatecall
     */
    function upgradeToAndCall(address newImplementation, bytes memory data) public payable virtual onlyProxy {
        _authorizeUpgrade(newImplementation);
        _upgradeToAndCallUUPS(newImplementation, data);
    }

    /**
     * @dev Reverts if the execution is not performed via delegatecall or the execution
     * context is not of a proxy with an ERC-1967 compliant implementation pointing to self.
     */
    function _checkProxy() internal view virtual {
        if (
            address(this) == __self || // Must be called through delegatecall
            ERC1967Utils.getImplementation() != __self // Must be called through an active proxy
        ) {
            revert UUPSUnauthorizedCallContext();
        }
    }

    /**
     * @dev Reverts if the execution is performed via delegatecall.
     * See {notDelegated}.
     */
    function _checkNotDelegated() internal view virtual {
        if (address(this) != __self) {
            // Must not be called through delegatecall
            revert UUPSUnauthorizedCallContext();
        }
    }

    /**
     * @dev Function that should revert when `msg.sender` is not authorized to upgrade the contract. Called by
     * {upgradeToAndCall}.
     *
     * Normally, this function will use an xref:access.adoc[access control] modifier such as {Ownable-onlyOwner}.
     *
     * ```solidity
     * function _authorizeUpgrade(address) internal onlyOwner {}
     * ```
     */
    function _authorizeUpgrade(address newImplementation) internal virtual;

    /**
     * @dev Performs an implementation upgrade with a security check for UUPS proxies, and additional setup call.
     *
     * As a security check, {proxiableUUID} is invoked in the new implementation, and the return value
     * is expected to be the implementation slot in ERC-1967.
     *
     * Emits an {IERC1967-Upgraded} event.
     */
    function _upgradeToAndCallUUPS(address newImplementation, bytes memory data) private {
        try IERC1822Proxiable(newImplementation).proxiableUUID() returns (bytes32 slot) {
            if (slot != ERC1967Utils.IMPLEMENTATION_SLOT) {
                revert UUPSUnsupportedProxiableUUID(slot);
            }
            ERC1967Utils.upgradeToAndCall(newImplementation, data);
        } catch {
            // The implementation is not UUPS
            revert ERC1967Utils.ERC1967InvalidImplementation(newImplementation);
        }
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/math/Math.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title Math
/// @notice OpenZeppelin Math 最小兼容库，本项目当前只需要全精度 mulDiv。
library Math {
    /// @notice 计算 floor(x * y / denominator)，支持 512 位中间乘积，避免 x * y 溢出。
    function mulDiv(uint256 x, uint256 y, uint256 denominator) internal pure returns (uint256 result) {
        unchecked {
            uint256 prod0;
            uint256 prod1;
            assembly {
                let mm := mulmod(x, y, not(0))
                prod0 := mul(x, y)
                prod1 := sub(sub(mm, prod0), lt(mm, prod0))
            }

            if (prod1 == 0) {
                require(denominator > 0, "Math: div0");
                return prod0 / denominator;
            }

            require(denominator > prod1, "Math: overflow");

            uint256 remainder;
            assembly {
                remainder := mulmod(x, y, denominator)
                prod1 := sub(prod1, gt(remainder, prod0))
                prod0 := sub(prod0, remainder)
            }

            uint256 twos = denominator & (0 - denominator);
            assembly {
                denominator := div(denominator, twos)
                prod0 := div(prod0, twos)
                twos := add(div(sub(0, twos), twos), 1)
            }
            prod0 |= prod1 * twos;

            uint256 inverse = (3 * denominator) ^ 2;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;

            result = prod0 * inverse;
            return result;
        }
    }
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/interfaces/draft-IERC1822.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (interfaces/draft-IERC1822.sol)

pragma solidity >=0.4.16;

/**
 * @dev ERC-1822: Universal Upgradeable Proxy Standard (UUPS) documents a method for upgradeability through a simplified
 * proxy whose upgrades are fully controlled by the current implementation.
 */
interface IERC1822Proxiable {
    /**
     * @dev Returns the storage slot that the proxiable contract assumes is being used to store the implementation
     * address.
     *
     * IMPORTANT: A proxy pointing at a proxiable contract should not be considered proxiable itself, because this risks
     * bricking a proxy that upgrades to it, by delegating to itself until out of gas. Thus it is critical that this
     * function revert if invoked through a proxy.
     */
    function proxiableUUID() external view returns (bytes32);
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/proxy/ERC1967/ERC1967Utils.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.6.0) (proxy/ERC1967/ERC1967Utils.sol)

pragma solidity ^0.8.21;

import {IBeacon} from "../beacon/IBeacon.sol";
import {IERC1967} from "../../interfaces/IERC1967.sol";
import {Address} from "../../utils/Address.sol";
import {StorageSlot} from "../../utils/StorageSlot.sol";

/**
 * @dev This library provides getters and event emitting update functions for
 * https://eips.ethereum.org/EIPS/eip-1967[ERC-1967] slots.
 */
library ERC1967Utils {
    /**
     * @dev Storage slot with the address of the current implementation.
     * This is the keccak-256 hash of "eip1967.proxy.implementation" subtracted by 1.
     */
    // solhint-disable-next-line private-vars-leading-underscore
    bytes32 internal constant IMPLEMENTATION_SLOT = 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc;

    /**
     * @dev The `implementation` of the proxy is invalid.
     */
    error ERC1967InvalidImplementation(address implementation);

    /**
     * @dev The `admin` of the proxy is invalid.
     */
    error ERC1967InvalidAdmin(address admin);

    /**
     * @dev The `beacon` of the proxy is invalid.
     */
    error ERC1967InvalidBeacon(address beacon);

    /**
     * @dev An upgrade function sees `msg.value > 0` that may be lost.
     */
    error ERC1967NonPayable();

    /**
     * @dev Returns the current implementation address.
     */
    function getImplementation() internal view returns (address) {
        return StorageSlot.getAddressSlot(IMPLEMENTATION_SLOT).value;
    }

    /**
     * @dev Stores a new address in the ERC-1967 implementation slot.
     */
    function _setImplementation(address newImplementation) private {
        if (newImplementation.code.length == 0) {
            revert ERC1967InvalidImplementation(newImplementation);
        }
        StorageSlot.getAddressSlot(IMPLEMENTATION_SLOT).value = newImplementation;
    }

    /**
     * @dev Performs implementation upgrade with additional setup call if data is nonempty.
     * This function is payable only if the setup call is performed, otherwise `msg.value` is rejected
     * to avoid stuck value in the contract.
     *
     * Emits an {IERC1967-Upgraded} event.
     */
    function upgradeToAndCall(address newImplementation, bytes memory data) internal {
        _setImplementation(newImplementation);
        emit IERC1967.Upgraded(newImplementation);

        if (data.length > 0) {
            Address.functionDelegateCall(newImplementation, data);
        } else {
            _checkNonPayable();
        }
    }

    /**
     * @dev Storage slot with the admin of the contract.
     * This is the keccak-256 hash of "eip1967.proxy.admin" subtracted by 1.
     */
    // solhint-disable-next-line private-vars-leading-underscore
    bytes32 internal constant ADMIN_SLOT = 0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103;

    /**
     * @dev Returns the current admin.
     *
     * TIP: To get this value clients can read directly from the storage slot shown below (specified by ERC-1967) using
     * the https://ethereum.org/developers/docs/apis/json-rpc/#eth_getstorageat[`eth_getStorageAt`] RPC call.
     * `0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103`
     */
    function getAdmin() internal view returns (address) {
        return StorageSlot.getAddressSlot(ADMIN_SLOT).value;
    }

    /**
     * @dev Stores a new address in the ERC-1967 admin slot.
     */
    function _setAdmin(address newAdmin) private {
        if (newAdmin == address(0)) {
            revert ERC1967InvalidAdmin(address(0));
        }
        StorageSlot.getAddressSlot(ADMIN_SLOT).value = newAdmin;
    }

    /**
     * @dev Changes the admin of the proxy.
     *
     * Emits an {IERC1967-AdminChanged} event.
     */
    function changeAdmin(address newAdmin) internal {
        emit IERC1967.AdminChanged(getAdmin(), newAdmin);
        _setAdmin(newAdmin);
    }

    /**
     * @dev The storage slot of the UpgradeableBeacon contract which defines the implementation for this proxy.
     * This is the keccak-256 hash of "eip1967.proxy.beacon" subtracted by 1.
     */
    // solhint-disable-next-line private-vars-leading-underscore
    bytes32 internal constant BEACON_SLOT = 0xa3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50;

    /**
     * @dev Returns the current beacon.
     */
    function getBeacon() internal view returns (address) {
        return StorageSlot.getAddressSlot(BEACON_SLOT).value;
    }

    /**
     * @dev Stores a new beacon in the ERC-1967 beacon slot.
     */
    function _setBeacon(address newBeacon) private {
        if (newBeacon.code.length == 0) {
            revert ERC1967InvalidBeacon(newBeacon);
        }

        StorageSlot.getAddressSlot(BEACON_SLOT).value = newBeacon;

        address beaconImplementation = IBeacon(newBeacon).implementation();
        if (beaconImplementation.code.length == 0) {
            revert ERC1967InvalidImplementation(beaconImplementation);
        }
    }

    /**
     * @dev Change the beacon and trigger a setup call if data is nonempty.
     * This function is payable only if the setup call is performed, otherwise `msg.value` is rejected
     * to avoid stuck value in the contract.
     *
     * Emits an {IERC1967-BeaconUpgraded} event.
     *
     * CAUTION: Invoking this function has no effect on an instance of {BeaconProxy} since v5, since
     * it uses an immutable beacon without looking at the value of the ERC-1967 beacon slot for
     * efficiency.
     */
    function upgradeBeaconToAndCall(address newBeacon, bytes memory data) internal {
        _setBeacon(newBeacon);
        emit IERC1967.BeaconUpgraded(newBeacon);

        if (data.length > 0) {
            Address.functionDelegateCall(IBeacon(newBeacon).implementation(), data);
        } else {
            _checkNonPayable();
        }
    }

    /**
     * @dev Reverts if `msg.value` is not zero. It can be used to avoid `msg.value` stuck in the contract
     * if an upgrade doesn't perform an initialization call.
     */
    function _checkNonPayable() private {
        if (msg.value > 0) {
            revert ERC1967NonPayable();
        }
    }
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/proxy/beacon/IBeacon.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (proxy/beacon/IBeacon.sol)

pragma solidity >=0.4.16;

/**
 * @dev This is the interface that {BeaconProxy} expects of its beacon.
 */
interface IBeacon {
    /**
     * @dev Must return an address that can be used as a delegate call target.
     *
     * {UpgradeableBeacon} will check that this address is a contract.
     */
    function implementation() external view returns (address);
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/interfaces/IERC1967.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (interfaces/IERC1967.sol)

pragma solidity >=0.4.11;

/**
 * @dev ERC-1967: Proxy Storage Slots. This interface contains the events defined in the ERC.
 */
interface IERC1967 {
    /**
     * @dev Emitted when the implementation is upgraded.
     */
    event Upgraded(address indexed implementation);

    /**
     * @dev Emitted when the admin account has changed.
     */
    event AdminChanged(address previousAdmin, address newAdmin);

    /**
     * @dev Emitted when the beacon is changed.
     */
    event BeaconUpgraded(address indexed beacon);
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/utils/Address.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.5.0) (utils/Address.sol)

pragma solidity ^0.8.20;

import {Errors} from "./Errors.sol";
import {LowLevelCall} from "./LowLevelCall.sol";

/**
 * @dev Collection of functions related to the address type
 */
library Address {
    /**
     * @dev There's no code at `target` (it is not a contract).
     */
    error AddressEmptyCode(address target);

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
     * https://solidity.readthedocs.io/en/v0.8.20/security-considerations.html#use-the-checks-effects-interactions-pattern[checks-effects-interactions pattern].
     */
    function sendValue(address payable recipient, uint256 amount) internal {
        if (address(this).balance < amount) {
            revert Errors.InsufficientBalance(address(this).balance, amount);
        }
        if (LowLevelCall.callNoReturn(recipient, amount, "")) {
            // call successful, nothing to do
            return;
        } else if (LowLevelCall.returnDataSize() > 0) {
            LowLevelCall.bubbleRevert();
        } else {
            revert Errors.FailedCall();
        }
    }

    /**
     * @dev Performs a Solidity function call using a low level `call`. A
     * plain `call` is an unsafe replacement for a function call: use this
     * function instead.
     *
     * If `target` reverts with a revert reason or custom error, it is bubbled
     * up by this function (like regular Solidity function calls). However, if
     * the call reverted with no returned reason, this function reverts with a
     * {Errors.FailedCall} error.
     *
     * Returns the raw returned data. To convert to the expected return value,
     * use https://solidity.readthedocs.io/en/latest/units-and-global-variables.html?highlight=abi.decode#abi-encoding-and-decoding-functions[`abi.decode`].
     *
     * Requirements:
     *
     * - `target` must be a contract.
     * - calling `target` with `data` must not revert.
     */
    function functionCall(address target, bytes memory data) internal returns (bytes memory) {
        return functionCallWithValue(target, data, 0);
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but also transferring `value` wei to `target`.
     *
     * Requirements:
     *
     * - the calling contract must have an ETH balance of at least `value`.
     * - the called Solidity function must be `payable`.
     */
    function functionCallWithValue(address target, bytes memory data, uint256 value) internal returns (bytes memory) {
        if (address(this).balance < value) {
            revert Errors.InsufficientBalance(address(this).balance, value);
        }
        bool success = LowLevelCall.callNoReturn(target, value, data);
        if (success && (LowLevelCall.returnDataSize() > 0 || target.code.length > 0)) {
            return LowLevelCall.returnData();
        } else if (success) {
            revert AddressEmptyCode(target);
        } else if (LowLevelCall.returnDataSize() > 0) {
            LowLevelCall.bubbleRevert();
        } else {
            revert Errors.FailedCall();
        }
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but performing a static call.
     */
    function functionStaticCall(address target, bytes memory data) internal view returns (bytes memory) {
        bool success = LowLevelCall.staticcallNoReturn(target, data);
        if (success && (LowLevelCall.returnDataSize() > 0 || target.code.length > 0)) {
            return LowLevelCall.returnData();
        } else if (success) {
            revert AddressEmptyCode(target);
        } else if (LowLevelCall.returnDataSize() > 0) {
            LowLevelCall.bubbleRevert();
        } else {
            revert Errors.FailedCall();
        }
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but performing a delegate call.
     */
    function functionDelegateCall(address target, bytes memory data) internal returns (bytes memory) {
        bool success = LowLevelCall.delegatecallNoReturn(target, data);
        if (success && (LowLevelCall.returnDataSize() > 0 || target.code.length > 0)) {
            return LowLevelCall.returnData();
        } else if (success) {
            revert AddressEmptyCode(target);
        } else if (LowLevelCall.returnDataSize() > 0) {
            LowLevelCall.bubbleRevert();
        } else {
            revert Errors.FailedCall();
        }
    }

    /**
     * @dev Tool to verify that a low level call to smart-contract was successful, and reverts if the target
     * was not a contract or bubbling up the revert reason (falling back to {Errors.FailedCall}) in case
     * of an unsuccessful call.
     *
     * NOTE: This function is DEPRECATED and may be removed in the next major release.
     */
    function verifyCallResultFromTarget(
        address target,
        bool success,
        bytes memory returndata
    ) internal view returns (bytes memory) {
        // only check if target is a contract if the call was successful and the return data is empty
        // otherwise we already know that it was a contract
        if (success && (returndata.length > 0 || target.code.length > 0)) {
            return returndata;
        } else if (success) {
            revert AddressEmptyCode(target);
        } else if (returndata.length > 0) {
            LowLevelCall.bubbleRevert(returndata);
        } else {
            revert Errors.FailedCall();
        }
    }

    /**
     * @dev Tool to verify that a low level call was successful, and reverts if it wasn't, either by bubbling the
     * revert reason or with a default {Errors.FailedCall} error.
     */
    function verifyCallResult(bool success, bytes memory returndata) internal pure returns (bytes memory) {
        if (success) {
            return returndata;
        } else if (returndata.length > 0) {
            LowLevelCall.bubbleRevert(returndata);
        } else {
            revert Errors.FailedCall();
        }
    }
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/utils/StorageSlot.sol =====
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


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/utils/Errors.sol =====
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


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/utils/LowLevelCall.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.6.0) (utils/LowLevelCall.sol)

pragma solidity ^0.8.20;

/**
 * @dev Library of low level call functions that implement different calling strategies to deal with the return data.
 *
 * WARNING: Using this library requires an advanced understanding of Solidity and how the EVM works. It is recommended
 * to use the {Address} library instead.
 */
library LowLevelCall {
    /// @dev Performs a Solidity function call using a low level `call` and ignoring the return data.
    function callNoReturn(address target, bytes memory data) internal returns (bool success) {
        return callNoReturn(target, 0, data);
    }

    /// @dev Same as {callNoReturn-address-bytes}, but allows specifying the value to be sent in the call.
    function callNoReturn(address target, uint256 value, bytes memory data) internal returns (bool success) {
        assembly ("memory-safe") {
            success := call(gas(), target, value, add(data, 0x20), mload(data), 0x00, 0x00)
        }
    }

    /// @dev Performs a Solidity function call using a low level `call` and returns the first 64 bytes of the result
    /// in the scratch space of memory. Useful for functions that return a tuple with two single-word values.
    ///
    /// WARNING: Do not assume that the results are zero if `success` is false. Memory can be already allocated
    /// and this function doesn't zero it out.
    function callReturn64Bytes(
        address target,
        bytes memory data
    ) internal returns (bool success, bytes32 result1, bytes32 result2) {
        return callReturn64Bytes(target, 0, data);
    }

    /// @dev Same as {callReturn64Bytes-address-bytes}, but allows specifying the value to be sent in the call.
    function callReturn64Bytes(
        address target,
        uint256 value,
        bytes memory data
    ) internal returns (bool success, bytes32 result1, bytes32 result2) {
        assembly ("memory-safe") {
            success := call(gas(), target, value, add(data, 0x20), mload(data), 0x00, 0x40)
            result1 := mload(0x00)
            result2 := mload(0x20)
        }
    }

    /// @dev Performs a Solidity function call using a low level `staticcall` and ignoring the return data.
    function staticcallNoReturn(address target, bytes memory data) internal view returns (bool success) {
        assembly ("memory-safe") {
            success := staticcall(gas(), target, add(data, 0x20), mload(data), 0x00, 0x00)
        }
    }

    /// @dev Performs a Solidity function call using a low level `staticcall` and returns the first 64 bytes of the result
    /// in the scratch space of memory. Useful for functions that return a tuple with two single-word values.
    ///
    /// WARNING: Do not assume that the results are zero if `success` is false. Memory can be already allocated
    /// and this function doesn't zero it out.
    function staticcallReturn64Bytes(
        address target,
        bytes memory data
    ) internal view returns (bool success, bytes32 result1, bytes32 result2) {
        assembly ("memory-safe") {
            success := staticcall(gas(), target, add(data, 0x20), mload(data), 0x00, 0x40)
            result1 := mload(0x00)
            result2 := mload(0x20)
        }
    }

    /// @dev Performs a Solidity function call using a low level `delegatecall` and ignoring the return data.
    function delegatecallNoReturn(address target, bytes memory data) internal returns (bool success) {
        assembly ("memory-safe") {
            success := delegatecall(gas(), target, add(data, 0x20), mload(data), 0x00, 0x00)
        }
    }

    /// @dev Performs a Solidity function call using a low level `delegatecall` and returns the first 64 bytes of the result
    /// in the scratch space of memory. Useful for functions that return a tuple with two single-word values.
    ///
    /// WARNING: Do not assume that the results are zero if `success` is false. Memory can be already allocated
    /// and this function doesn't zero it out.
    function delegatecallReturn64Bytes(
        address target,
        bytes memory data
    ) internal returns (bool success, bytes32 result1, bytes32 result2) {
        assembly ("memory-safe") {
            success := delegatecall(gas(), target, add(data, 0x20), mload(data), 0x00, 0x40)
            result1 := mload(0x00)
            result2 := mload(0x20)
        }
    }

    /// @dev Returns the size of the return data buffer.
    function returnDataSize() internal pure returns (uint256 size) {
        assembly ("memory-safe") {
            size := returndatasize()
        }
    }

    /// @dev Returns a buffer containing the return data from the last call.
    function returnData() internal pure returns (bytes memory result) {
        assembly ("memory-safe") {
            result := mload(0x40)
            mstore(result, returndatasize())
            returndatacopy(add(result, 0x20), 0x00, returndatasize())
            mstore(0x40, add(result, add(0x20, returndatasize())))
        }
    }

    /// @dev Revert with the return data from the last call.
    function bubbleRevert() internal pure {
        assembly ("memory-safe") {
            let fmp := mload(0x40)
            returndatacopy(fmp, 0x00, returndatasize())
            revert(fmp, returndatasize())
        }
    }

    function bubbleRevert(bytes memory returndata) internal pure {
        assembly ("memory-safe") {
            revert(add(returndata, 0x20), mload(returndata))
        }
    }
}
