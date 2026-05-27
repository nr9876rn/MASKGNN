// ===== FILE: src/factory/UpgradeableVaultFactory.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { CoreAccess } from "../base/CoreAccess.sol";
import { Blocklist } from "../access/Blocklist.sol";
import { CoreVault } from "../vault/CoreVault.sol";
import { FeeManager } from "../modules/FeeManager.sol";
import { NAVOracle } from "../modules/NAVOracle.sol";
import { RebalanceManager } from "../modules/RebalanceManager.sol";
import { RequestManager } from "../modules/RequestManager.sol";
import { ShareToken } from "../tokens/ShareToken.sol";
import { VaultAuthority } from "../access/VaultAuthority.sol";
import { Whitelist } from "../access/Whitelist.sol";
import { Roles } from "../libraries/Roles.sol";
import { ERC1967Proxy } from "oz-v5/proxy/ERC1967/ERC1967Proxy.sol";
import { AssetCustodian } from "../modules/AssetCustodian.sol";
import { OndoTarget } from "../interfaces/Types.sol";

/// @title UpgradeableVaultFactory
/// @notice UUPS Proxy Factory: Each module deploys only a lightweight ERC1967Proxy, and the business logic reuses the implementation.
contract UpgradeableVaultFactory is CoreAccess {
    struct Deployment {
        address authority;
        address whitelist;
        address blocklist;
        address shareToken;
        address navOracle;
        address feeManager;
        address requestManager;
        address rebalanceManager;
        address coreVault;
        address assetCustodian;
    }

    struct Implementations {
        address whitelistImpl;
        address blocklistImpl;
        address shareTokenImpl;
        address navOracleImpl;
        address feeManagerImpl;
        address requestManagerImpl;
        address rebalanceManagerImpl;
        address coreVaultImpl;
        address assetCustodianImpl;
    }

    struct VaultConfig {
        uint256 navStaleAfter;
        uint256 maxRoundJump;
        uint256 annualManagementFeeBps;
        uint256 performanceFeeBps;
        uint256 hurdleBps;
        uint256 initialHighWaterMark;
        uint256 minRedeemAssets;
        uint16 defaultSlippageBps;
        uint16 defaultLossPercentBps;
        uint256 minDeposit;
        uint256 maxDepositAmount;
        uint256 tvlCap;
    }

    Implementations public implementations;

    event ImplementationsSet(Implementations implementations);
    event VaultDeployed(
        address indexed coreVault,
        address indexed assetCustodian,
        address indexed shareToken,
        address authority,
        address superAdmin,
        address admin
    );

    constructor(address authority_) CoreAccess(authority_) { }

    function setImplementations(Implementations calldata implementations_) external onlyAdmin {
        _setImplementations(implementations_);
    }

    function deployVault(
        string memory vaultName,
        string memory vaultSymbol,
        string memory shareName,
        string memory shareSymbol,
        address usdc,
        address ondoProxy,
        address trustedSigner,
        address superAdmin,
        OndoTarget[] calldata targets
    ) external onlyAdmin returns (Deployment memory d) {
        d = _deployVault(
            vaultName,
            vaultSymbol,
            shareName,
            shareSymbol,
            usdc,
            ondoProxy,
            trustedSigner,
            superAdmin,
            targets,
            _defaultVaultConfig()
        );
        emit VaultDeployed(d.coreVault, d.assetCustodian, d.shareToken, d.authority, superAdmin, msg.sender);
    }

    function deployVaultWithConfig(
        string memory vaultName,
        string memory vaultSymbol,
        string memory shareName,
        string memory shareSymbol,
        address usdc,
        address ondoProxy,
        address trustedSigner,
        address superAdmin,
        OndoTarget[] calldata targets,
        VaultConfig calldata config
    ) external onlyAdmin returns (Deployment memory d) {
        d = _deployVault(
            vaultName,
            vaultSymbol,
            shareName,
            shareSymbol,
            usdc,
            ondoProxy,
            trustedSigner,
            superAdmin,
            targets,
            _validateVaultConfig(config)
        );
        emit VaultDeployed(d.coreVault, d.assetCustodian, d.shareToken, d.authority, superAdmin, msg.sender);
    }

    function _deployVault(
        string memory vaultName,
        string memory vaultSymbol,
        string memory shareName,
        string memory shareSymbol,
        address usdc,
        address ondoProxy,
        address trustedSigner,
        address superAdmin,
        OndoTarget[] calldata targets,
        VaultConfig memory config
    ) internal returns (Deployment memory d) {
        require(superAdmin != address(0), "UpgradeableFactory: zero super admin");
        d.authority = address(new VaultAuthority(superAdmin, msg.sender));
        VaultAuthority(d.authority).grantRole(Roles.CURATOR, msg.sender);
        VaultAuthority(d.authority).grantRole(Roles.OPERATOR, msg.sender);
        VaultAuthority(d.authority).grantRole(Roles.MONITOR, msg.sender);
        VaultAuthority(d.authority).grantRole(Roles.PAUSE_ROLE, msg.sender);

        d.whitelist = _proxy(implementations.whitelistImpl, abi.encodeCall(Whitelist.initialize, (d.authority)));
        d.blocklist = _proxy(implementations.blocklistImpl, abi.encodeCall(Blocklist.initialize, (d.authority)));
        d.shareToken = _proxy(
            implementations.shareTokenImpl,
            abi.encodeCall(ShareToken.initialize, (shareName, shareSymbol, d.authority, d.blocklist))
        );
        d.navOracle = _proxy(
            implementations.navOracleImpl,
            abi.encodeCall(NAVOracle.initialize, (d.authority, config.navStaleAfter, config.maxRoundJump))
        );
        d.feeManager = _proxy(
            implementations.feeManagerImpl,
            abi.encodeCall(
                FeeManager.initialize,
                (
                    d.authority,
                    d.shareToken,
                    config.annualManagementFeeBps,
                    config.performanceFeeBps,
                    config.hurdleBps,
                    config.initialHighWaterMark
                )
            )
        );
        d.requestManager =
            _proxy(implementations.requestManagerImpl, abi.encodeCall(RequestManager.initialize, (d.authority)));
        d.rebalanceManager = _proxy(
            implementations.rebalanceManagerImpl, abi.encodeCall(RebalanceManager.initialize, (d.authority, usdc))
        );
        d.coreVault = _proxy(
            implementations.coreVaultImpl,
            abi.encodeCall(
                CoreVault.initialize, (d.authority, vaultName, vaultSymbol, d.shareToken, usdc, d.navOracle, d.whitelist, d.blocklist)
            )
        );
        d.assetCustodian = _proxy(
            implementations.assetCustodianImpl,
            abi.encodeCall(
                AssetCustodian.initialize, (d.authority, d.coreVault, ondoProxy, usdc, d.navOracle)
            )
        );
        // 写配置信息
        _wire(d, usdc, trustedSigner, superAdmin, targets, config);
        VaultAuthority(d.authority).renounceRole(Roles.DEFAULT_ADMIN_ROLE, address(this));
    }

    function _proxy(address implementation, bytes memory data) internal returns (address) {
        return address(new ERC1967Proxy(implementation, data));
    }

    function _wire(
        Deployment memory d,
        address usdc,
        address trustedSigner,
        address ondoTargetManager,
        OndoTarget[] calldata targets,
        VaultConfig memory config
    ) internal {
        VaultAuthority(d.authority).grantRole(Roles.FEE_CONTROLLER, d.coreVault);
        VaultAuthority(d.authority).grantRole(Roles.TOKEN_MINTER, d.feeManager);
        VaultAuthority(d.authority).grantRole(Roles.NAV_CALLER, d.coreVault);
        VaultAuthority(d.authority).grantRole(Roles.NAV_CALLER, d.assetCustodian);
        VaultAuthority(d.authority).grantRole(Roles.ONDO_TARGET_MANAGER, address(this));
        VaultAuthority(d.authority).grantRole(Roles.ONDO_TARGET_MANAGER, ondoTargetManager);

        ShareToken(d.shareToken).addVault(usdc, d.coreVault);
        CoreVault(d.coreVault).setAc(d.assetCustodian);
        CoreVault(d.coreVault).setFeeManager(d.feeManager);
        RequestManager(d.requestManager).setCoreVault(d.coreVault);
        RebalanceManager(d.rebalanceManager).setCoreVault(d.coreVault);
        CoreVault(d.coreVault).setRequestManager(d.requestManager);
        CoreVault(d.coreVault).setRebalanceManager(d.rebalanceManager);
        NAVOracle(d.navOracle).setCoreVault(d.coreVault);
        NAVOracle(d.navOracle).setAuthorizedCaller(d.coreVault, true);
        NAVOracle(d.navOracle).setAuthorizedCaller(d.assetCustodian, true);
        NAVOracle(d.navOracle).setTrustedSigner(trustedSigner, true);
        CoreVault(d.coreVault).setOndoTargets(targets);
        VaultAuthority(d.authority).revokeRole(Roles.ONDO_TARGET_MANAGER, address(this));
        CoreVault(d.coreVault).setRiskParams(config.minDeposit, config.maxDepositAmount, config.tvlCap, config.minRedeemAssets);
        CoreVault(d.coreVault).setDefaultSlippageBps(config.defaultSlippageBps);
        CoreVault(d.coreVault).setDefaultLossPercentBps(config.defaultLossPercentBps);
    }

    function _defaultVaultConfig() internal pure returns (VaultConfig memory) {
        return VaultConfig({
            navStaleAfter: 48 hours,
            maxRoundJump: 1000,
            annualManagementFeeBps: 95,
            performanceFeeBps: 2000,
            hurdleBps: 1288,
            initialHighWaterMark: 1e18,
            minRedeemAssets: 1e6,
            defaultSlippageBps: 100,
            defaultLossPercentBps: 100,
            minDeposit: 1e6,
            maxDepositAmount: 1_000_000e6,
            tvlCap: 10_000_000e18
        });
    }

    function _validateVaultConfig(VaultConfig calldata config) internal pure returns (VaultConfig memory cfg) {
        require(config.navStaleAfter > 0, "UpgradeableFactory: zero staleness");
        require(config.maxRoundJump > 0, "UpgradeableFactory: zero round jump");
        require(config.annualManagementFeeBps <= 2000, "UpgradeableFactory: management too high");
        require(config.performanceFeeBps <= 5000, "UpgradeableFactory: performance too high");
        require(config.hurdleBps <= 10_000, "UpgradeableFactory: bad hurdle");
        require(config.initialHighWaterMark > 0, "UpgradeableFactory: zero hwm");
        require(config.defaultSlippageBps <= 2000, "UpgradeableFactory: slippage too high");
        require(config.defaultLossPercentBps <= 2000, "UpgradeableFactory: lossPercent too high");
        require(config.minDeposit <= config.maxDepositAmount, "UpgradeableFactory: min > max deposit");
        require(config.maxDepositAmount > 0, "UpgradeableFactory: zero maxDeposit");
        require(config.tvlCap > 0, "UpgradeableFactory: zero tvlCap");
        return config;
    }

    function _setImplementations(Implementations memory implementations_) internal {
        require(
            implementations_.whitelistImpl.code.length > 0 && implementations_.blocklistImpl.code.length > 0
                && implementations_.shareTokenImpl.code.length > 0 && implementations_.navOracleImpl.code.length > 0
                && implementations_.feeManagerImpl.code.length > 0
                && implementations_.requestManagerImpl.code.length > 0
                && implementations_.rebalanceManagerImpl.code.length > 0
                && implementations_.coreVaultImpl.code.length > 0
                && implementations_.assetCustodianImpl.code.length > 0,
            "UpgradeableFactory: bad impl"
        );
        implementations = implementations_;
        emit ImplementationsSet(implementations_);
    }
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


// ===== FILE: src/access/Blocklist.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { CoreAccess } from "../base/CoreAccess.sol";
import { IBlocklist } from "../interfaces/IBlocklist.sol";
import { Roles } from "../libraries/Roles.sol";

/// @title Blocklist
contract Blocklist is CoreAccess, IBlocklist {
    mapping(address => bool) private _blocked;

    event BlocklistSet(address indexed account, bool blocked);
    event BlocklistReasonSet(address indexed account, bool blocked, string reason);

    constructor(address admin) CoreAccess(admin) { }

    function initialize(address admin) external {
        _initializeCoreAccess(admin);
    }

    function setBlocked(address account, bool blocked) external onlyRole(Roles.DEFAULT_ADMIN_ROLE) {
        _blocked[account] = blocked;
        emit BlocklistSet(account, blocked);
    }

    function setBlockedWithReason(address account, bool blocked, string calldata reason)
        external
        onlyRole(Roles.DEFAULT_ADMIN_ROLE)
    {
        _blocked[account] = blocked;
        emit BlocklistSet(account, blocked);
        emit BlocklistReasonSet(account, blocked, reason);
    }

    function setBlockedBatch(address[] calldata accounts, bool[] calldata blocked)
        external
        onlyRole(Roles.DEFAULT_ADMIN_ROLE)
    {
        require(accounts.length == blocked.length, "Blocklist: length mismatch");
        for (uint256 i; i < accounts.length; ++i) {
            _blocked[accounts[i]] = blocked[i];
            emit BlocklistSet(accounts[i], blocked[i]);
        }
    }

    function isBlocked(address account) external view returns (bool) {
        return _blocked[account];
    }
}


// ===== FILE: src/vault/CoreVault.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { IERC20 } from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import { IERC20Metadata } from "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import { SafeERC20 } from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import { Pausable } from "@openzeppelin/contracts/utils/Pausable.sol";
import { ReentrancyGuard } from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import { IERC165 } from "@openzeppelin/contracts/utils/introspection/IERC165.sol";
import { CoreAccess } from "../base/CoreAccess.sol";
import { IAssetCustodian } from "../interfaces/IAssetCustodian.sol";
import { IBlocklist } from "../interfaces/IBlocklist.sol";
import { IFeeManager } from "../interfaces/IFeeManager.sol";
import { IERC7575 } from "../interfaces/IERC7575.sol";
import { INAVOracle } from "../interfaces/INAVOracle.sol";
import { IRebalanceManager } from "../interfaces/IRebalanceManager.sol";
import { IRequestManager } from "../interfaces/IRequestManager.sol";
import { IShareToken } from "../interfaces/IShareToken.sol";
import { IWhitelist } from "../interfaces/IWhitelist.sol";
import { IVaultSettlement } from "../interfaces/IVaultSettlement.sol";
import {
    BatchHint,
    OndoTarget,
    PassiveRebalanceParams,
    RebalanceBuy,
    RebalanceTrigger,
    SignedPriceBundle,
    Trade
} from "../interfaces/Types.sol";
import { FixedPoint } from "../libraries/FixedPoint.sol";
import { Roles } from "../libraries/Roles.sol";

/// @title CoreVault
contract CoreVault is CoreAccess, Pausable, ReentrancyGuard, IVaultSettlement, IERC7575 {
    using SafeERC20 for IERC20;

    uint256 internal constant MAX_STOCK_TOKENS = 20;

    enum VaultStatus {
        Normal,
        Rebalancing,
        Paused
    }

    IShareToken public shareToken;
    IERC20 public usdc;
    INAVOracle public navOracle;
    IWhitelist public whitelist;
    IBlocklist public blocklist;
    IFeeManager public feeManager;
    IRequestManager public requestManager;
    IRebalanceManager public rebalanceManager;
    IAssetCustodian public ac;
    VaultStatus public vaultStatus;
    uint256 public minRedeemAssets;
    uint16 public defaultSlippageBps;
    uint16 public defaultLossPercentBps;

    uint256 public minDeposit;
    uint256 public maxDepositAmount;
    uint256 public tvlCap;
    uint256 public completedDepositAssets;
    uint256 public pendingDepositAssets;

    uint256 public latestTotalAssetsValue;
    mapping(uint256 => uint256) public totalAssetsByRound;

    address[] public _stockTokens;
    mapping(address => uint256) public targetWeightBps;

    bool public whitelistRequired;

    string public vaultName;
    string public vaultSymbol;

    /// @notice Subscription fee rate, in bps (per ten thousand), for example, 50 = 0.5%.
    uint16 public depositFeeBps;
    address public feeRecipient;

    bool private _coreVaultInitialized;

    event ACSet(address indexed ac);
    event FeeManagerSet(address indexed feeManager);
    event RequestManagerSet(address indexed requestManager);
    event RebalanceManagerSet(address indexed rebalanceManager);
    event Deposit(address indexed sender, address indexed owner, uint256 assets, uint256 shares);
    event Withdraw(
        address indexed sender, address indexed receiver, address indexed owner, uint256 assets, uint256 shares
    );
    event DepositRequested(
        uint256 indexed reqId, address indexed owner, uint256 assets, uint256 shares, bytes32 groupId
    );
    event RedeemRequested(uint256 indexed reqId, address indexed owner, uint256 shares, bytes32 groupId);
    event DepositFulfilled(uint256 indexed reqId, address indexed owner, uint256 shares, uint256 refundUsdc);
    event RedeemFulfilled(uint256 indexed reqId, address indexed owner, uint256 shares, uint256 usdcReceived);
    event RequestFailed(uint256 indexed reqId, bool deposit, bytes reason);
    event OndoTargetsSet(bytes32 indexed allocationHash);
    event RebalanceRequested(bytes32 indexed rbId, bytes32 indexed groupId);
    event RebalanceFailed(bytes32 indexed rbId, bytes reason);
    event PassiveRebalanceRequested(
        bytes32 indexed rbId,
        bytes32 indexed groupId,
        RebalanceTrigger indexed trigger,
        uint8 riskLevel,
        bytes32 reasonHash
    );
    event RebalanceFulfilled(bytes32 indexed rbId, uint256 totalAssetsAfter);
    event TotalAssetsUpdated(uint256 indexed round, uint256 totalAssetsValue);
    event VaultForcedNormal(address indexed caller);

    modifier onlyAc() {
        _onlyAc();
        _;
    }

    modifier onlyNormalVault() {
        _onlyNormalVault();
        _;
    }

    function supportsInterface(bytes4 interfaceId) external pure returns (bool) {
        return interfaceId == type(IERC165).interfaceId || interfaceId == type(IERC7575).interfaceId;
    }

    function share() external view returns (address shareTokenAddress) {
        return address(shareToken);
    }

    function asset() external view returns (address assetTokenAddress) {
        return address(usdc);
    }

    constructor(
        address admin,
        string memory vaultName_,
        string memory vaultSymbol_,
        address shareToken_,
        address usdc_,
        address navOracle_,
        address whitelist_,
        address blocklist_
    ) CoreAccess(admin) {
        if (admin != address(0)) {
            _initializeCoreVault(vaultName_, vaultSymbol_, shareToken_, usdc_, navOracle_, whitelist_, blocklist_);
        }
    }

    function initialize(
        address admin,
        string memory vaultName_,
        string memory vaultSymbol_,
        address shareToken_,
        address usdc_,
        address navOracle_,
        address whitelist_,
        address blocklist_
    ) external {
        _initializeCoreAccess(admin);
        _initializeCoreVault(vaultName_, vaultSymbol_, shareToken_, usdc_, navOracle_, whitelist_, blocklist_);
    }

    function _initializeCoreVault(
        string memory vaultName_,
        string memory vaultSymbol_,
        address shareToken_,
        address usdc_,
        address navOracle_,
        address whitelist_,
        address blocklist_
    ) internal {
        require(!_coreVaultInitialized, "CV:INIT");
        require(shareToken_ != address(0) && usdc_ != address(0) && navOracle_ != address(0), "CV:ZA");
        _coreVaultInitialized = true;
        vaultName = vaultName_;
        vaultSymbol = vaultSymbol_;
        shareToken = IShareToken(shareToken_);
        usdc = IERC20(usdc_);
        navOracle = INAVOracle(navOracle_);
        whitelist = IWhitelist(whitelist_);
        blocklist = IBlocklist(blocklist_);
    }

    function setAc(address ac_) external onlyAdmin {
        require(ac_ != address(0), "CV:ZA");
        ac = IAssetCustodian(ac_);
        emit ACSet(ac_);
    }

    function setFeeManager(address feeManager_) external onlyAdmin {
        feeManager = IFeeManager(feeManager_);
        emit FeeManagerSet(feeManager_);
    }

    function setRequestManager(address requestManager_) external onlyAdmin {
        require(requestManager_ != address(0), "CV:ZA");
        requestManager = IRequestManager(requestManager_);
        emit RequestManagerSet(requestManager_);
    }

    function setRebalanceManager(address rebalanceManager_) external onlyAdmin {
        require(rebalanceManager_ != address(0), "CV:ZA");
        rebalanceManager = IRebalanceManager(rebalanceManager_);
        emit RebalanceManagerSet(rebalanceManager_);
    }

    function setRiskParams(uint256 minDeposit_, uint256 maxDeposit_, uint256 tvlCap_, uint256 minRedeemAssets_) external onlyAdmin {
        require(minDeposit_ <= maxDeposit_, "CV:DR");
        minDeposit = minDeposit_;
        maxDepositAmount = maxDeposit_;
        minRedeemAssets = minRedeemAssets_;
        tvlCap = tvlCap_;
    }

    function setDefaultSlippageBps(uint16 slippageBps) external onlyAdmin {
        require(slippageBps <= 2000, "CV:SL");
        defaultSlippageBps = slippageBps;
    }

    function setDefaultLossPercentBps(uint16 lossPercentBps) external onlyAdmin {
        require(lossPercentBps <= 10_000, "CV:LP");
        defaultLossPercentBps = lossPercentBps;
    }

    function setWhitelistRequired(bool required) external onlyAdmin {
        whitelistRequired = required;
    }

    function setFeeParams(uint16 depositFeeBps_, address feeRecipient_) external onlyAdmin {
        require(feeRecipient_ != address(0), "CV:ZA");
        depositFeeBps = depositFeeBps_;
        feeRecipient = feeRecipient_;
    }

    function setOndoTargets(OndoTarget[] calldata targets) external onlyRole(Roles.ONDO_TARGET_MANAGER) {
        _validateOndoTargets(targets);
        for (uint256 i; i < _stockTokens.length; ++i) {
            targetWeightBps[_stockTokens[i]] = 0;
        }
        delete _stockTokens;
        for (uint256 i; i < targets.length; ++i) {
            _stockTokens.push(targets[i].token);
            targetWeightBps[targets[i].token] = targets[i].bps;
        }
        emit OndoTargetsSet(_ondoAllocationHash());
    }

    function pause() external onlyRole(Roles.PAUSE_ROLE) {
        vaultStatus = VaultStatus.Paused;
        _pause();
    }

    function unpause() external onlyRole(Roles.PAUSE_ROLE) {
        _unpause();
        if (address(rebalanceManager) != address(0) && rebalanceManager.activeRebalanceId() != bytes32(0)) {
            vaultStatus = VaultStatus.Rebalancing;
        } else {
            vaultStatus = VaultStatus.Normal;
        }
    }

    function forceNormal() external onlyAdmin {
        if (paused()) _unpause();
        vaultStatus = VaultStatus.Normal;
        emit VaultForcedNormal(msg.sender);
    }

    /// @notice User initiates USDC purchase.
    /// @dev Process: Verify price bundle signature -> Calculate expected Share -> Retrieve user's USDC -> Transfer to AC -> AC calls Proxy.
    /// @param assets Quantity of USDC purchased, 6-digit precision.
    /// @param owner Source of funds and final Share recipient.
    /// @param priceBundle Price bundle signed by the trusted signer.
    /// @param minSharesOut Minimum acceptable shares for slippage protection.
    /// @return reqId ID of this asynchronous request.
    /// @return feeAmount USDC subscription fee charged for this request.
    function requestDeposit(
        uint256 assets,
        address owner,
        SignedPriceBundle calldata priceBundle,
        uint256 minSharesOut
    )
        external
        whenNotPaused
        onlyNormalVault
        nonReentrant
        returns (uint256 reqId, uint256 feeAmount)
    {
        uint64 round = navOracle.ingest(priceBundle);
        _updateTotalAssetsForRound(round);
        _accrueManagementFeeIfConfigured();
        return _requestDeposit(assets, owner, round, minSharesOut);
    }

    /// @notice User initiates a share redemption.
    // @dev Process: Verify price bundle signature -> Lock user Share -> Transfer underlying assets to AC
    // according to share -> AC calls Proxy Sell.
    // @param shares Share token amount to redeem, 18-digit precision.
    // @param owner Source of Shares and final USDC recipient.
    // @param priceBundle Price bundle signed by the trusted signer.
    // @return reqId ID of this asynchronous request.
    function requestRedeem(uint256 shares, address owner, SignedPriceBundle calldata priceBundle)
        external
        whenNotPaused
        onlyNormalVault
        nonReentrant
        returns (uint256 reqId)
    {
        uint64 round = navOracle.ingest(priceBundle);
        _updateTotalAssetsForRound(round);
        _accrueManagementFeeIfConfigured();
        return _requestRedeem(shares, owner, round);
    }


    /// @notice Curator active rebalancing entry point. Uses the same two-phase sell/buy flow as passive rebalancing.
    function requestRebalance(
        Trade[] calldata sells,
        RebalanceBuy[] calldata buys,
        address[] calldata targetTokens,
        uint256[] calldata targetWeights,
        SignedPriceBundle calldata priceBundle
    )
        external
        onlyRole(Roles.CURATOR)
        whenNotPaused
        onlyNormalVault
        nonReentrant
        returns (bytes32 rbId, bytes32 groupId)
    {
        require(address(rebalanceManager) != address(0), "CV:RB0");
        uint64 round = navOracle.ingest(priceBundle);
        _updateTotalAssetsForRound(round);
        _accrueManagementFeeIfConfigured();
        rbId = rebalanceManager.prepareActiveRebalance(sells.length, buys.length, buys, targetTokens, targetWeights);
        _savePendingOndoTargets(rbId, targetTokens, targetWeights);
        vaultStatus = VaultStatus.Rebalancing;
        groupId = _dispatchPassiveRebalance(rbId, sells, buys, RebalanceTrigger.Active);
        require(groupId != bytes32(0), "CV:ZB");
        emit RebalanceRequested(rbId, groupId);
    }

    /// @notice Monitor Service Passive portfolio rebalancing entry point.
    function requestPassiveRebalance(PassiveRebalanceParams calldata params)
        external
        onlyRole(Roles.MONITOR)
        whenNotPaused
        onlyNormalVault
        nonReentrant
        returns (bytes32 rbId, bytes32 groupId)
    {
        require(address(rebalanceManager) != address(0), "CV:RB0");
        uint64 round = navOracle.ingest(params.priceBundle);
        _updateTotalAssetsForRound(round);
        _accrueManagementFeeIfConfigured();
        rbId = rebalanceManager.preparePassiveRebalance(
            params.sells.length,
            params.buys.length,
            params.trigger,
            params.riskLevel,
            params.reasonHash,
            params.buys,
            params.targetTokens,
            params.targetWeights
        );
        _savePendingOndoTargets(rbId, params.targetTokens, params.targetWeights);
        vaultStatus = VaultStatus.Rebalancing;
        // Distribute and adjust stacks to reduce the stack depth of individual functions.
        groupId = _dispatchPassiveRebalance(rbId, params.sells, params.buys, params.trigger);
        require(groupId != bytes32(0), "CV:ZB");
        emit PassiveRebalanceRequested(rbId, groupId, params.trigger, params.riskLevel, params.reasonHash);
    }

    /// --------------------------------- Settlement Entry Points (Called by AC) -----------------------------------
    function settleDeposit(
        bytes32 groupId,
        uint256 reqId,
        address[] calldata etfTokens,
        uint256[] calldata amounts,
        uint256 refundUsdc,
        uint256 navRound
    ) external onlyAc nonReentrant {
        require(etfTokens.length == amounts.length, "CV:LEN");
        _validateSettlementNavRound(navRound);
        IRequestManager.DepositSettlement memory result =
            requestManager.settleDeposit(groupId, reqId, refundUsdc);
        pendingDepositAssets -= result.spent + result.refund;
        completedDepositAssets += result.spent;
        if (result.shares > 0) {
            shareToken.mint(result.owner, result.shares);
            if (address(feeManager) != address(0)) feeManager.startAccrual();
            emit DepositFulfilled(reqId, result.owner, result.shares, 0);
        }
    }

    function settleRedeem(bytes32 groupId, uint256 reqId, uint256 usdcReceived, uint256 navRound)
        external
        onlyAc
        nonReentrant
    {
        _validateSettlementNavRound(navRound);
        IRequestManager.RedeemSettlement memory result =
            requestManager.settleRedeem(groupId, reqId, usdcReceived);
        if (result.fulfilledShares > 0) {
            shareToken.burn(address(this), result.fulfilledShares);
        }
        if (result.payout > 0) {
            usdc.safeTransfer(result.owner, result.payout);
        }
        emit RedeemFulfilled(reqId, result.owner, result.fulfilledShares, result.payout);
    }

    function settleRebalance(bytes32 rbId, address[] calldata tokens, uint256[] calldata amounts)
        external
        onlyAc
        nonReentrant
    {
        require(tokens.length == amounts.length, "CV:LEN");
        uint256 round = navOracle.latestRound();
        uint256 totalAssetsAfter = _updateTotalAssetsForRound(round);
        vaultStatus = paused() ? VaultStatus.Paused : VaultStatus.Normal;
        rebalanceManager.settleRebalanceComplete(rbId, tokens, amounts);
        _applyPendingOndoTargets(rbId);
        emit RebalanceFulfilled(rbId, totalAssetsAfter);
    }

    function settleRebalanceFailure(bytes32 rbId, bytes calldata reason) external onlyAc nonReentrant {
        vaultStatus = paused() ? VaultStatus.Paused : VaultStatus.Normal;
        rebalanceManager.settleRebalanceFailed(rbId);
        _clearPendingOndoTargets(rbId);
        emit RebalanceFailed(rbId, reason);
    }

    function settleFailure(bytes32 groupId, uint256 reqId, uint256 returnedUsdc, bytes calldata reason)
        external
        onlyAc
        nonReentrant
    {
        IRequestManager.RequestData memory request = requestManager.getRequest(reqId);
        IRequestManager.FailureSettlement memory result =
            requestManager.settleRequestFailure(groupId, reqId, returnedUsdc);
        if (result.isDeposit) {
            pendingDepositAssets -= request.assets;
            if (result.refund > 0) usdc.safeTransfer(result.owner, result.refund);
            emit RequestFailed(reqId, true, reason);
        } else {
            IERC20(address(shareToken)).safeTransfer(result.owner, result.shares);
            emit RequestFailed(reqId, false, reason);
        }
    }

    /// ---------------------------Fee Calculate---------------------------------

    /// @notice Settle dividend fees with the latest signed price package
    function settlePerformanceFee(SignedPriceBundle calldata priceBundle)
        external
        onlyRole(Roles.OPERATOR)
        nonReentrant
        returns (uint256)
    {
        require(vaultStatus == VaultStatus.Normal, "CV:NN");
        uint64 round = navOracle.ingest(priceBundle);
        _updateTotalAssetsForRound(round);
        _accrueManagementFeeIfConfigured();
        if (address(feeManager) == address(0)) return 0;
        uint256 totalAssetsValue = _totalAssetsForRound(round);
        (uint256 shares,) = feeManager.settlePerformanceFee(totalAssetsValue, effectiveTotalSupply());
        return shares;
    }

    /// @dev Lazy loading supplements management fees
    function _accrueManagementFeeIfConfigured() internal returns (uint256 shares) {
        if (address(feeManager) == address(0)) return 0;
        uint256 supply = effectiveTotalSupply();
        if (supply == 0) return 0;
        shares = feeManager.accrueManagementFee(totalAssets(), supply);
    }

    ///  ---------------------------Public Function---------------------------------

    function getTargetAllocation() public view returns (address[] memory tokens, uint256[] memory weights) {
        tokens = new address[](_stockTokens.length);
        weights = new uint256[](_stockTokens.length);
        for (uint256 i; i < _stockTokens.length; ++i) {
            tokens[i] = _stockTokens[i];
            weights[i] = targetWeightBps[_stockTokens[i]];
        }
    }

    function totalAssets() public view returns (uint256) {
        return latestTotalAssetsValue;
    }

    /// @notice After updating the price with the latest signed price package, return the net asset value per share (NAV) with 18 digits precision.
    // @dev NAV = totalAssetsValue / effectiveTotalSupply. The initial NAV is 1e18 (i.e., 1 USDC/share).
    function navPerShare(SignedPriceBundle calldata priceBundle) public returns (uint256) {
        uint64 round = navOracle.ingest(priceBundle);
        uint256 totalAssetsValue = _updateTotalAssetsForRound(round);
        uint256 supply = effectiveTotalSupply() - requestManager.escrowedShares();
        if (supply == 0) return 1e18;
        return totalAssetsValue * 1e18 / supply;
    }

    function totalSupply() public view returns (uint256) {
        return IERC20(address(shareToken)).totalSupply();
    }

    /// @notice Returns the effective total share used for valuation, which is the actual total supply plus the share of uncrystallized costs.
    function effectiveTotalSupply() public view returns (uint256) {
        if (address(feeManager) == address(0)) return totalSupply();
        return feeManager.effectiveTotalSupply(totalSupply());
    }

    function convertToShares(uint256 assets, SignedPriceBundle calldata priceBundle) external view returns (uint256) {
        return _previewDeposit(
            FixedPoint.usdcToWad(assets),
            navOracle.recomputeTotalAssetsFromBundle(address(this), address(usdc), priceBundle),
            effectiveTotalSupply() - requestManager.escrowedShares()
        );
    }

    function previewDepositWithFee(uint256 assets, SignedPriceBundle calldata priceBundle)
        external
        view
        returns (uint256 shares, uint256 feeAmount)
    {
        feeAmount = assets * depositFeeBps / 10_000;
        uint256 netAssets = assets - feeAmount;
        shares = _previewDeposit(
            FixedPoint.usdcToWad(netAssets),
            navOracle.recomputeTotalAssetsFromBundle(address(this), address(usdc), priceBundle),
            effectiveTotalSupply() - requestManager.escrowedShares()
        );
    }

    function convertToAssets(uint256 shares, SignedPriceBundle calldata priceBundle) external view returns (uint256) {
        uint256 supply = effectiveTotalSupply() - requestManager.escrowedShares();
        if (supply == 0) return 0;
        return FixedPoint.wadToUsdc(
            shares * navOracle.recomputeTotalAssetsFromBundle(address(this), address(usdc), priceBundle) / supply
        );
    }

    ///  ---------------------------Private Function---------------------------------

    function _requestDeposit(uint256 assets, address owner, uint64 round, uint256 minSharesOut)
        internal
        returns (uint256 reqId, uint256 feeAmount)
    {
        require(address(ac) != address(0), "CV:AC0");
        require(address(requestManager) != address(0), "CV:RM0");
        require(assets >= minDeposit && assets <= maxDepositAmount, "CV:DB");
        _checkAccount(owner);
        require(msg.sender == owner, "CV:OP");
        // Deposit Fee
        feeAmount = assets * depositFeeBps / 10_000;
        uint256 netAssets = assets - feeAmount;
        require(totalAssets() + FixedPoint.usdcToWad(pendingDepositAssets + netAssets) <= tvlCap, "CV:CAP");
        if (feeAmount > 0 && feeRecipient != address(0)) {
            usdc.safeTransferFrom(owner, feeRecipient, feeAmount);
        }
        usdc.safeTransferFrom(owner, address(ac), netAssets);
        pendingDepositAssets += netAssets;
        // execution
        uint256 totalAssetsValue = _totalAssetsForRound(round);
        uint256 supply = effectiveTotalSupply() - requestManager.escrowedShares();
        uint256 shares = _previewDeposit(FixedPoint.usdcToWad(netAssets), totalAssetsValue, supply);
        require(shares >= minSharesOut, "CV:MS");
        reqId = requestManager.createDepositRequest(owner, netAssets, shares, round, totalAssetsValue, supply);
        bytes32 expectedGroupId = ac.previewNextGroupId();
        requestManager.setGroupId(reqId, expectedGroupId);
        bytes32 groupId = ac.executeDeposit(reqId, netAssets, _defaultBatchHint());
        require(groupId == expectedGroupId, "CV:GB");
        emit Deposit(msg.sender, owner, assets, shares);
        emit DepositRequested(reqId, owner, netAssets, shares, groupId);
    }

    function _requestRedeem(uint256 shares, address owner, uint64 round)
        internal
        returns (uint256 reqId)
    {
        require(address(ac) != address(0), "CV:AC0");
        require(address(requestManager) != address(0), "CV:RM0");
        require(shares > 0, "CV:ZS");
        _checkAccount(owner);
        require(msg.sender == owner, "CV:OP");
        uint256 totalAssetsValue = _totalAssetsForRound(round);
        uint256 availableSupply = effectiveTotalSupply() - requestManager.escrowedShares();
        require(totalAssetsValue > 0 && availableSupply > 0, "CV:ZR");
        require(shares <= availableSupply, "CV:RS");
        uint256 expectedAssets = shares * totalAssetsValue / availableSupply;
        require(minRedeemAssets == 0 || expectedAssets >= FixedPoint.usdcToWad(minRedeemAssets), "CV:BRA");
        reqId = requestManager.createRedeemRequest(owner, shares, round, totalAssetsValue, availableSupply, expectedAssets);
        bytes32 expectedGroupId = ac.previewNextGroupId();
        requestManager.setGroupId(reqId, expectedGroupId);
        IERC20(address(shareToken)).safeTransferFrom(owner, address(this), shares);
        bytes32 groupId = _dispatchRedeem(reqId, shares, availableSupply);
        require(groupId == expectedGroupId, "CV:GB");
        emit Withdraw(msg.sender, owner, owner, 0, shares);
        emit RedeemRequested(reqId, owner, shares, groupId);
    }

    function _dispatchRedeem(uint256 reqId, uint256 shares, uint256 supply)
        internal
        returns (bytes32 groupId)
    {
        (address[] memory tokens, uint256[] memory amounts) = _transferRedeemBasketToAc(shares, supply);
        groupId = ac.executeRedeem(reqId, tokens, amounts, _defaultBatchHint());
    }

    function _dispatchPassiveRebalance(
        bytes32 rbId,
        Trade[] calldata sells,
        RebalanceBuy[] calldata buys,
        RebalanceTrigger trigger
    ) internal returns (bytes32 groupId) {
        _transferRebalanceSellsToAc(sells);
        groupId = ac.executeRebalance(rbId, sells, buys, trigger, _defaultBatchHint());
    }

    function _transferRebalanceSellsToAc(Trade[] calldata sells) internal {
        for (uint256 i; i < sells.length; ++i) {
            require(sells[i].amountIn > 0, "CV:ZS");
            IERC20(sells[i].tokenIn).safeTransfer(address(ac), sells[i].amountIn);
        }
    }

    function _transferRedeemBasketToAc(uint256 shares, uint256 supply)
        internal
        returns (address[] memory tokens, uint256[] memory amounts)
    {
        uint256 count = _stockTokens.length;
        tokens = new address[](count);
        amounts = new uint256[](count);
        if (supply == 0) return (tokens, amounts);
        for (uint256 i; i < count; ++i) {
            address token = _stockTokens[i];
            tokens[i] = token;
            uint256 bal = IERC20(token).balanceOf(address(this));
            uint256 amt = bal * shares / supply;
            amounts[i] = amt;
            if (amt > 0) IERC20(token).safeTransfer(address(ac), amt);
        }
    }

    function _defaultBatchHint() internal view returns (BatchHint memory) {
        return BatchHint({ maxSlippageBps: defaultSlippageBps, maxLossPercentBps: defaultLossPercentBps });
    }


    function _setOndoTargetsFromPending(bytes32 rbId) internal {
        OndoTarget[] memory pending = rebalanceManager.getPendingOndoTargets(rbId);
        for (uint256 i; i < _stockTokens.length; ++i) {
            targetWeightBps[_stockTokens[i]] = 0;
        }
        delete _stockTokens;
        for (uint256 i; i < pending.length; ++i) {
            _stockTokens.push(pending[i].token);
            targetWeightBps[pending[i].token] = pending[i].bps;
        }
        emit OndoTargetsSet(_ondoAllocationHash());
    }

    function _savePendingOndoTargets(
        bytes32 rbId,
        address[] calldata targetTokens,
        uint256[] calldata targetWeights
    ) internal {
        require(targetTokens.length <= MAX_STOCK_TOKENS, "CV:TL");
        rebalanceManager.savePendingOndoTargets(rbId, targetTokens, targetWeights);
    }

    function _applyPendingOndoTargets(bytes32 rbId) internal {
        _setOndoTargetsFromPending(rbId);
        _clearPendingOndoTargets(rbId);
    }

    function _clearPendingOndoTargets(bytes32 rbId) internal {
        rebalanceManager.clearPendingOndoTargets(rbId);
    }

    function _validateOndoTargets(OndoTarget[] calldata targets) internal pure {
        require(targets.length <= MAX_STOCK_TOKENS, "CV:TL");
        uint256 total;
        for (uint256 i; i < targets.length; ++i) {
            require(targets[i].token != address(0), "CV:ZA");
            for (uint256 j; j < i; ++j) {
                require(targets[i].token != targets[j].token, "CV:DUP");
            }
            total += targets[i].bps;
        }
        require(total == 10_000, "CV:BT");
    }

    function _ondoAllocationHash() internal view returns (bytes32) {
        // forge-lint: disable-next-line(asm-keccak256)
        return keccak256(abi.encode(_stockTokens, _targetWeights()));
    }

    function _onlyAc() internal view {
        require(msg.sender == address(ac), "CV:AC");
    }

    function _onlyNormalVault() internal view {
        require(vaultStatus == VaultStatus.Normal, "CV:NN");
    }

    function _checkAccount(address account) internal view {
        require(account != address(0), "CV:ZA");
        require(!blocklist.isBlocked(account), "CV:BLK");
        if (whitelistRequired) require(whitelist.isWhitelisted(account), "CV:WL");
    }

    function _recomputedTotalAssetsFromOraclePrices() internal view returns (uint256 totalAssetsValue) {
        totalAssetsValue = FixedPoint.usdcToWad(usdc.balanceOf(address(this)));
        uint256 count = _stockTokens.length;
        for (uint256 i; i < count; ++i) {
            address token = _stockTokens[i];
            if (token == address(usdc)) continue;
            uint256 balance = IERC20(token).balanceOf(address(this));
            if (balance == 0) continue;
            uint256 price = navOracle.priceOf(token);
            require(price > 0, "CV:MP");
            totalAssetsValue += balance * price / _assetUnit(token);
        }
    }

    function _targetWeights() internal view returns (uint256[] memory weights) {
        weights = new uint256[](_stockTokens.length);
        for (uint256 i; i < _stockTokens.length; ++i) {
            weights[i] = targetWeightBps[_stockTokens[i]];
        }
    }

    function _validateSettlementNavRound(uint256 navRound) internal view {
        require(navRound != 0 && navRound <= navOracle.latestRound(), "CV:INR");
    }

    function _assetUnit(address token) internal view returns (uint256) {
        uint8 decimals = IERC20Metadata(token).decimals();
        require(decimals <= 36, "CV:DEC");
        return 10 ** uint256(decimals);
    }

    function _updateTotalAssetsForRound(uint256 round) internal returns (uint256 totalAssetsValue) {
        require(round != 0 && round <= navOracle.latestRound(), "CV:INR");
        totalAssetsValue = _recomputedTotalAssetsFromOraclePrices();
        latestTotalAssetsValue = totalAssetsValue;
        totalAssetsByRound[round] = totalAssetsValue;
        emit TotalAssetsUpdated(round, totalAssetsValue);
    }

    function _totalAssetsForRound(uint256 round) internal view returns (uint256 totalAssetsValue) {
        totalAssetsValue = totalAssetsByRound[round];
        require(totalAssetsValue > 0 || totalSupply() == 0, "CV:ZN");
    }

    /// User's share = Invested Amount / Net Value per Share
    // = Invested Amount / (totalAssetsValue / totalSupply)
    // = Invested Amount * totalSupply / totalAssetsValue
    function _previewDeposit(uint256 assets18, uint256 totalAssetsValue_, uint256 supply) internal pure returns (uint256) {
        if (supply == 0 || totalAssetsValue_ == 0) return assets18;
        return assets18 * supply / totalAssetsValue_;
    }

}


// ===== FILE: src/modules/FeeManager.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { CoreAccess } from "../base/CoreAccess.sol";
import { IFeeManager } from "../interfaces/IFeeManager.sol";
import { IShareToken } from "../interfaces/IShareToken.sol";
import { FixedPoint } from "../libraries/FixedPoint.sol";
import { Roles } from "../libraries/Roles.sol";

/// @title FeeManager
/// @notice Management fee/dividend fee module; currently implements lazy loading of management fees based on time difference and clears the accounts when crystallization occurs.
contract FeeManager is CoreAccess, IFeeManager {
    uint256 public constant INITIAL_SHARE_PRICE = 1e18;
    IShareToken public shareToken;
    uint256 public annualManagementFeeBps;
    uint256 public performanceFeeBps;
    uint256 public hurdleBps;
    uint256 public highWaterMark;
    uint256 public lastPerformanceSettleTimestamp;
    uint256 public performanceSettleInterval;
    uint256 public unclaimedManagementFeeShares;
    uint256 public claimedManagementFeeShares;
    uint256 public unclaimedPerformanceFeeShares;
    uint256 public claimedPerformanceFeeShares;
    uint256 public lastAccrualTimestamp;
    bool private _feeManagerInitialized;

    event AccrualStarted(uint256 indexed timestamp);
    event ManagementFeeAccrued(uint256 indexed fromTimestamp, uint256 indexed toTimestamp, uint256 shares);
    event ManagementFeeClaimed(address indexed recipient, uint256 shares);
    event PerformanceFeeClaimed(address indexed recipient, uint256 shares);
    event PerformanceFeeSettled(
        uint256 shares,
        uint256 currentSharePrice,
        uint256 threshold,
        uint256 newHighWaterMark
    );
    event FeeConfigSet(uint256 annualManagementFeeBps, uint256 performanceFeeBps, uint256 hurdleBps);
    event PerformanceSettleIntervalSet(uint256 interval);

    constructor(
        address admin,
        address shareToken_,
        uint256 annualManagementFeeBps_,
        uint256 performanceFeeBps_,
        uint256 hurdleBps_,
        uint256 highWaterMark_
    ) CoreAccess(admin) {
        if (admin != address(0)) {
            _initializeFeeManager(
                shareToken_, annualManagementFeeBps_, performanceFeeBps_, hurdleBps_, highWaterMark_
            );
        }
    }

    function initialize(
        address admin,
        address shareToken_,
        uint256 annualManagementFeeBps_,
        uint256 performanceFeeBps_,
        uint256 hurdleBps_,
        uint256 highWaterMark_
    ) external {
        _initializeCoreAccess(admin);
        _initializeFeeManager(
            shareToken_, annualManagementFeeBps_, performanceFeeBps_, hurdleBps_, highWaterMark_
        );
    }

    function _initializeFeeManager(
        address shareToken_,
        uint256 annualManagementFeeBps_,
        uint256 performanceFeeBps_,
        uint256 hurdleBps_,
        uint256 highWaterMark_
    ) internal {
        require(!_feeManagerInitialized, "FeeManager: initialized");
        require(shareToken_ != address(0), "FeeManager: zero address");
        _feeManagerInitialized = true;
        shareToken = IShareToken(shareToken_);
        annualManagementFeeBps = annualManagementFeeBps_;
        performanceFeeBps = performanceFeeBps_;
        hurdleBps = hurdleBps_;
        highWaterMark = highWaterMark_;
    }

    function setFeeConfig(uint256 annualBps, uint256 performanceBps, uint256 hurdleBps_) external onlyAdmin {
        require(annualBps <= 2000 && performanceBps <= 5000, "FeeManager: fee too high");
        require(hurdleBps_ <= FixedPoint.BPS, "FeeManager: hurdle too high");
        annualManagementFeeBps = annualBps;
        performanceFeeBps = performanceBps;
        hurdleBps = hurdleBps_;
        emit FeeConfigSet(annualBps, performanceBps, hurdleBps_);
    }

    function setPerformanceSettleInterval(uint256 interval) external onlyAdmin {
        performanceSettleInterval = interval;
        emit PerformanceSettleIntervalSet(interval);
    }

    function startAccrual() external onlyRole(Roles.FEE_CONTROLLER) {
        if (lastAccrualTimestamp == 0) {
            lastAccrualTimestamp = block.timestamp;
            emit AccrualStarted(block.timestamp);
        }
    }

    function accrueManagementFee(uint256 totalAssetsValue, uint256 effectiveSupply)
        public
        onlyRole(Roles.FEE_CONTROLLER)
        returns (uint256 shares)
    {
        if (lastAccrualTimestamp == 0) {
            lastAccrualTimestamp = block.timestamp;
            emit AccrualStarted(block.timestamp);
            return 0;
        }
        uint256 fromTimestamp = lastAccrualTimestamp;
        uint256 elapsed = block.timestamp - fromTimestamp;
        lastAccrualTimestamp = block.timestamp;

        if (elapsed == 0 || totalAssetsValue == 0 || effectiveSupply == 0 || annualManagementFeeBps == 0) {
            emit ManagementFeeAccrued(fromTimestamp, block.timestamp, 0);
            return 0;
        }
        // Management fees are calculated as "Total Assets Value × Annualized Fee Rate × Elapsed Time as a Percentage of the Annual Total".
        // Example: totalAssetsValue = 10,000,000 USDC, Annualized Management Fee = 95 bps = 0.95%, elapsed = 1 day,
        // feeAssets = 10,000,000 * 0.95% * 1 / 365, representing the asset value for which management fees should be collected in 1 day.
        uint256 feeAssets = totalAssetsValue * annualManagementFeeBps * elapsed / FixedPoint.BPS / 365 days;
        shares = _feeSharesFromAssets(feeAssets, totalAssetsValue, effectiveSupply);
        unclaimedManagementFeeShares += shares;
        emit ManagementFeeAccrued(fromTimestamp, block.timestamp, shares);
    }

    function claimManagementFee() public onlyRole(Roles.CURATOR) returns (uint256 shares) {
        shares = unclaimedManagementFeeShares;
        unclaimedManagementFeeShares = 0;
        claimedManagementFeeShares += shares;
        if (shares > 0) shareToken.mint(msg.sender, shares);
        emit ManagementFeeClaimed(msg.sender, shares);
    }

    function claimPerformanceFee() public onlyRole(Roles.CURATOR) returns (uint256 shares) {
        shares = unclaimedPerformanceFeeShares;
        unclaimedPerformanceFeeShares = 0;
        claimedPerformanceFeeShares += shares;
        if (shares > 0) shareToken.mint(msg.sender, shares);
        emit PerformanceFeeClaimed(msg.sender, shares);
    }

    function claimAllFees() external onlyRole(Roles.CURATOR) returns (uint256 shares) {
        shares = unclaimedManagementFeeShares + unclaimedPerformanceFeeShares;
        if (unclaimedManagementFeeShares > 0) {
            claimedManagementFeeShares += unclaimedManagementFeeShares;
            emit ManagementFeeClaimed(msg.sender, unclaimedManagementFeeShares);
            unclaimedManagementFeeShares = 0;
        }
        if (unclaimedPerformanceFeeShares > 0) {
            claimedPerformanceFeeShares += unclaimedPerformanceFeeShares;
            emit PerformanceFeeClaimed(msg.sender, unclaimedPerformanceFeeShares);
            unclaimedPerformanceFeeShares = 0;
        }
        if (shares > 0) shareToken.mint(msg.sender, shares);
    }

    function previewPerformanceFee(uint256 totalAssetsValue, uint256 effectiveSupply)
        public
        view
        returns (uint256 currentSharePrice, uint256 threshold, uint256 feeShares)
    {
        if (totalAssetsValue == 0 || effectiveSupply == 0 || performanceFeeBps == 0) {
            return (0, _performanceThreshold(), 0);
        }
        currentSharePrice = _sharePrice(totalAssetsValue, effectiveSupply);
        threshold = _performanceThreshold();
        if (currentSharePrice <= threshold) return (currentSharePrice, threshold, 0);

        uint256 excessPerShare = currentSharePrice - threshold;
        uint256 totalExcessAssets = excessPerShare * effectiveSupply / 1e18;
        uint256 feeAssets = totalExcessAssets * performanceFeeBps / FixedPoint.BPS;
        feeShares = _feeSharesFromAssets(feeAssets, totalAssetsValue, effectiveSupply);
    }

    function settlePerformanceFee(uint256 totalAssetsValue, uint256 effectiveSupply)
        external
        onlyRole(Roles.FEE_CONTROLLER)
        returns (uint256 shares, uint256 newHighWaterMark)
    {
        if (lastPerformanceSettleTimestamp != 0 && performanceSettleInterval > 0) {
            require(
                block.timestamp >= lastPerformanceSettleTimestamp + performanceSettleInterval,
                "FeeManager: performance too soon"
            );
        }
        (uint256 currentSharePrice, uint256 threshold, uint256 feeShares) =
            previewPerformanceFee(totalAssetsValue, effectiveSupply);
        shares = feeShares;
        unclaimedPerformanceFeeShares += shares;

        if (currentSharePrice > highWaterMark) {
            highWaterMark = currentSharePrice;
        }
        newHighWaterMark = highWaterMark;

        if (shares > 0) {
            lastPerformanceSettleTimestamp = block.timestamp;
        }

        emit PerformanceFeeSettled(shares, currentSharePrice, threshold, newHighWaterMark);
    }

    function effectiveTotalSupply(uint256 totalSupply) external view returns (uint256) {
        return totalSupply + uncollectedFeeInShares();
    }

    function uncollectedFeeInShares() public view returns (uint256) {
        return unclaimedManagementFeeShares + unclaimedPerformanceFeeShares;
    }

    /// @dev max(highWaterMark, initialSharePrice × (1 + hurdleBps))。
    function _performanceThreshold() internal view returns (uint256 threshold) {
        uint256 hurdleSharePrice = INITIAL_SHARE_PRICE * (FixedPoint.BPS + hurdleBps) / FixedPoint.BPS;
        threshold = highWaterMark > hurdleSharePrice ? highWaterMark : hurdleSharePrice;
    }

    function _sharePrice(uint256 totalAssetsValue, uint256 effectiveSupply) internal pure returns (uint256) {
        return totalAssetsValue * 1e18 / effectiveSupply;
    }

    function _feeSharesFromAssets(uint256 feeAssets, uint256 totalAssetsValue, uint256 effectiveSupply)
        internal
        pure
        returns (uint256)
    {
        if (feeAssets == 0) return 0;
        require(feeAssets < totalAssetsValue, "FeeManager: fee exceeds assets");
        return feeAssets * effectiveSupply / (totalAssetsValue - feeAssets);
    }
}


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


// ===== FILE: src/modules/RebalanceManager.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { CoreAccess } from "../base/CoreAccess.sol";
import { IRebalanceManager } from "../interfaces/IRebalanceManager.sol";
import { RebalanceBuy, RebalanceTrigger, OndoTarget } from "../interfaces/Types.sol";

/// @title RebalanceManager
/// @notice saves the rebalancing status, target position, and transaction verification data, but does not directly hold or transfer assets.
contract RebalanceManager is CoreAccess, IRebalanceManager {
    address public coreVault;
    address public usdc;
    uint256 public rebalanceNonce;
    uint256 public minPassiveRebalanceInterval;
    uint256 public lastPassiveRebalanceAt;
    bytes32 public activeRebalanceId;
    mapping(bytes32 => address[]) private _rebalanceExpectedBuyTokens;
    mapping(bytes32 => uint256[]) private _rebalanceExpectedMinAmounts;
    mapping(bytes32 => RebalanceTrigger) public rebalanceTrigger;
    mapping(bytes32 => uint8) public rebalanceRiskLevel;
    mapping(bytes32 => bytes32) public rebalanceReasonHash;
    mapping(bytes32 => bool) public settledGroup;
    mapping(bytes32 => OndoTarget[]) private _pendingOndoTargets;
    bool private _rebalanceManagerInitialized;

    modifier onlyCoreVault() {
        _onlyCoreVault();
        _;
    }

    constructor(address admin, address usdc_) CoreAccess(admin) {
        if (admin != address(0)) _initializeRebalanceManager(usdc_);
    }

    function initialize(address admin, address usdc_) external {
        _initializeCoreAccess(admin);
        _initializeRebalanceManager(usdc_);
    }

    function _initializeRebalanceManager(address usdc_) internal {
        require(!_rebalanceManagerInitialized, "RebalanceManager: initialized");
        require(usdc_ != address(0), "RebalanceManager: zero usdc");
        _rebalanceManagerInitialized = true;
        usdc = usdc_;
    }

    function setCoreVault(address coreVault_) external onlyAdmin {
        require(coreVault_ != address(0), "RebalanceManager: zero core");
        coreVault = coreVault_;
    }

    function setPassiveRebalanceInterval(uint256 interval) external onlyAdmin {
        minPassiveRebalanceInterval = interval;
    }

    function prepareActiveRebalance(
        uint256 sellCount,
        uint256 buyCount,
        RebalanceBuy[] calldata buys,
        address[] calldata targetTokens,
        uint256[] calldata targetWeights
    ) external onlyCoreVault returns (bytes32 rbId) {
        require(activeRebalanceId == bytes32(0), "RebalanceManager: active");
        require(sellCount > 0 && buyCount > 0, "RebalanceManager: empty active");
        rbId = _newRebalanceId(sellCount, buyCount, RebalanceTrigger.Active);
        rebalanceTrigger[rbId] = RebalanceTrigger.Active;
        _saveExpectedBuys(rbId, buys);
        _validateTargets(targetTokens, targetWeights);
        activeRebalanceId = rbId;
    }

    function preparePassiveRebalance(
        uint256 sellCount,
        uint256 buyCount,
        RebalanceTrigger trigger,
        uint8 riskLevel,
        bytes32 reasonHash,
        RebalanceBuy[] calldata buys,
        address[] calldata targetTokens,
        uint256[] calldata targetWeights
    ) external onlyCoreVault returns (bytes32 rbId) {
        require(activeRebalanceId == bytes32(0), "RebalanceManager: active");
        _validatePassive(targetTokens.length, targetWeights.length, trigger);
        rbId = _newRebalanceId(sellCount, buyCount, trigger);
        lastPassiveRebalanceAt = block.timestamp;
        rebalanceTrigger[rbId] = trigger;
        rebalanceRiskLevel[rbId] = riskLevel;
        rebalanceReasonHash[rbId] = reasonHash;
        _validateTargets(targetTokens, targetWeights);
        _saveExpectedBuys(rbId, buys);
        activeRebalanceId = rbId;
    }

    function validateRebalanceFilled(bytes32 rbId, address[] calldata tokens, uint256[] calldata amounts)
        external
        view
    {
        _validateRebalanceFilled(rbId, tokens, amounts);
    }

    function _validateRebalanceFilled(bytes32 rbId, address[] calldata tokens, uint256[] calldata amounts)
        internal
        view
    {
        address[] storage expectedTokens = _rebalanceExpectedBuyTokens[rbId];
        uint256[] storage expectedAmounts = _rebalanceExpectedMinAmounts[rbId];
        for (uint256 i; i < expectedTokens.length; ++i) {
            uint256 actualAmount;
            for (uint256 j; j < tokens.length; ++j) {
                if (tokens[j] == expectedTokens[i]) actualAmount += amounts[j];
            }
            require(actualAmount >= expectedAmounts[i], "RebalanceManager: partial fill");
        }
    }

    function settleRebalance(bytes32 rbId) external onlyCoreVault {
        _settleRebalanceInternal(rbId);
    }

    function settleRebalanceFailure(bytes32 rbId) external onlyCoreVault {
        _settleRebalanceFailureInternal(rbId);
    }

    function _settleRebalanceInternal(bytes32 rbId) internal {
        _clearExpectedBuys(rbId);
        activeRebalanceId = bytes32(0);
    }

    function _settleRebalanceFailureInternal(bytes32 rbId) internal {
        _clearExpectedBuys(rbId);
        activeRebalanceId = bytes32(0);
    }

    ///  ---------------------------Settlement Entry Points---------------------------------

    function settleRebalanceComplete(
        bytes32 rbId,
        address[] calldata tokens,
        uint256[] calldata amounts
    ) external onlyCoreVault {
        if (settledGroup[rbId]) return;
        require(tokens.length == amounts.length, "RebalanceManager: length mismatch");
        _validateRebalanceFilled(rbId, tokens, amounts);
        settledGroup[rbId] = true;
        _settleRebalanceInternal(rbId);
        // Note: pendingOndoTargets will be applied by CoreVault after this call
    }

    function settleRebalanceFailed(bytes32 rbId) external onlyCoreVault {
        if (settledGroup[rbId]) return;
        settledGroup[rbId] = true;
        _settleRebalanceFailureInternal(rbId);
        _clearPendingOndoTargets(rbId);
    }

    function isSettled(bytes32 rbId) external view returns (bool) {
        return settledGroup[rbId];
    }

    ///  ---------------------------Pending Ondo Targets Management---------------------------------

    function savePendingOndoTargets(
        bytes32 rbId,
        address[] calldata targetTokens,
        uint256[] calldata targetWeights
    ) external onlyCoreVault {
        require(targetTokens.length == targetWeights.length, "RebalanceManager: length mismatch");
        delete _pendingOndoTargets[rbId];
        uint256 total;
        for (uint256 i; i < targetTokens.length; ++i) {
            require(targetTokens[i] != address(0), "RebalanceManager: zero target");
            require(targetWeights[i] <= type(uint16).max, "RebalanceManager: bad weight");
            for (uint256 j; j < i; ++j) {
                require(targetTokens[i] != targetTokens[j], "RebalanceManager: duplicate");
            }
            total += targetWeights[i];
            _pendingOndoTargets[rbId].push(OndoTarget({ token: targetTokens[i], bps: uint16(targetWeights[i]) }));
        }
        require(total == 10_000, "RebalanceManager: bad total");
    }

    function getPendingOndoTargets(bytes32 rbId) external view returns (OndoTarget[] memory) {
        return _pendingOndoTargets[rbId];
    }

    function clearPendingOndoTargets(bytes32 rbId) external onlyCoreVault {
        _clearPendingOndoTargets(rbId);
    }

    function _clearPendingOndoTargets(bytes32 rbId) internal {
        delete _pendingOndoTargets[rbId];
    }

    function _validatePassive(uint256 targetTokenLength, uint256 targetWeightLength, RebalanceTrigger trigger)
        internal
        view
    {
        require(
            trigger == RebalanceTrigger.PassivePeriodic || trigger == RebalanceTrigger.PassiveRiskEvent,
            "RebalanceManager: bad trigger"
        );
        require(targetTokenLength == targetWeightLength, "RebalanceManager: length mismatch");
        if (trigger == RebalanceTrigger.PassivePeriodic && minPassiveRebalanceInterval > 0) {
            require(
                block.timestamp >= lastPassiveRebalanceAt + minPassiveRebalanceInterval, "RebalanceManager: too soon"
            );
        }
    }

    function _newRebalanceId(uint256 sellCount, uint256 buyCount, RebalanceTrigger trigger) internal returns (bytes32) {
        return _rebalanceHash(coreVault, block.chainid, ++rebalanceNonce, sellCount, buyCount, trigger);
    }

    function _onlyCoreVault() internal view {
        require(msg.sender == coreVault, "RebalanceManager: not core");
    }

    function _rebalanceHash(
        address vault,
        uint256 chainId,
        uint256 nonce_,
        uint256 sellCount,
        uint256 buyCount,
        RebalanceTrigger trigger
    ) internal pure returns (bytes32 result) {
        assembly ("memory-safe") {
            let ptr := mload(0x40)
            mstore(ptr, and(vault, 0xffffffffffffffffffffffffffffffffffffffff))
            mstore(add(ptr, 0x20), chainId)
            mstore(add(ptr, 0x40), 0xe0)
            mstore(add(ptr, 0x60), nonce_)
            mstore(add(ptr, 0x80), sellCount)
            mstore(add(ptr, 0xa0), buyCount)
            mstore(add(ptr, 0xc0), trigger)
            mstore(add(ptr, 0xe0), 2)
            mstore(add(ptr, 0x100), shl(240, 0x5242))
            result := keccak256(ptr, 0x120)
        }
    }

    function _validateTargets(address[] calldata targetTokens, uint256[] calldata targetWeights) internal pure {
        require(targetTokens.length == targetWeights.length, "RebalanceManager: length mismatch");
        uint256 total;
        for (uint256 i; i < targetTokens.length; ++i) {
            require(targetTokens[i] != address(0), "RebalanceManager: zero target");
            for (uint256 j; j < i; ++j) {
                require(targetTokens[i] != targetTokens[j], "RebalanceManager: duplicate");
            }
            total += targetWeights[i];
        }
        require(total == 10_000, "RebalanceManager: bad weights");
    }

    function _saveExpectedBuys(bytes32 rbId, RebalanceBuy[] calldata buys) internal {
        for (uint256 i; i < buys.length; ++i) {
            require(buys[i].tokenIn != address(0), "RebalanceManager: zero buy tokenIn");
            require(buys[i].tokenOut != address(0), "RebalanceManager: zero buy tokenOut");
            require(buys[i].tokenIn == usdc, "RebalanceManager: buy tokenIn");
            require(buys[i].amountIn > 0, "RebalanceManager: zero amount");
            _rebalanceExpectedBuyTokens[rbId].push(buys[i].tokenOut);
            _rebalanceExpectedMinAmounts[rbId].push(buys[i].minAmountOut);
        }
    }

    function _clearExpectedBuys(bytes32 rbId) internal {
        delete _rebalanceExpectedBuyTokens[rbId];
        delete _rebalanceExpectedMinAmounts[rbId];
    }
}


// ===== FILE: src/modules/RequestManager.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { CoreAccess } from "../base/CoreAccess.sol";
import { IRequestManager } from "../interfaces/IRequestManager.sol";
import { RequestStatus } from "../interfaces/Types.sol";
import { FixedPoint } from "../libraries/FixedPoint.sol";

/// @title RequestManager
/// @notice Saves the status of users' asynchronous subscription/redemption orders, but does not hold any funds.
contract RequestManager is CoreAccess, IRequestManager {
    address public coreVault;
    uint256 public nextRequestId = 1;
    mapping(uint256 => RequestData) private _requests;
    mapping(address => uint256[]) private _userRequestIds;
    mapping(uint256 => uint256) public requestExpectedValue;
    mapping(address => uint256) public userEscrowedShares;
    uint256 public escrowedShares;
    mapping(bytes32 => bool) public settledGroup;
    bool private _requestManagerInitialized;

    modifier onlyCoreVault() {
        _onlyCoreVault();
        _;
    }

    constructor(address admin) CoreAccess(admin) { }

    function initialize(address admin) external {
        _initializeCoreAccess(admin);
        require(!_requestManagerInitialized, "RequestManager: initialized");
        _requestManagerInitialized = true;
        nextRequestId = 1;
    }

    function setCoreVault(address coreVault_) external onlyAdmin {
        require(coreVault_ != address(0), "RequestManager: zero core");
        coreVault = coreVault_;
    }

    ///  ---------------------------Order Creation---------------------------------

    function createDepositRequest(
        address owner,
        uint256 assets,
        uint256 shares,
        uint256 navRound,
        uint256 totalAssetsSnapshot,
        uint256 supplySnapshot
    ) external onlyCoreVault returns (uint256 reqId) {
        reqId = nextRequestId++;
        _requests[reqId] = RequestData({
            owner: owner,
            isDeposit: true,
            assets: assets,
            shares: shares,
            settledShares: 0,
            refundAssets: 0,
            navRound: navRound,
            totalAssetsSnapshot: totalAssetsSnapshot,
            supplySnapshot: supplySnapshot,
            status: RequestStatus.Executing,
            groupId: bytes32(0)
        });
        _userRequestIds[owner].push(reqId);
    }


    function createRedeemRequest(
        address owner,
        uint256 shares,
        uint256 navRound,
        uint256 totalAssetsSnapshot,
        uint256 supplySnapshot,
        uint256 expectedAssets
    ) external onlyCoreVault returns (uint256 reqId) {
        reqId = nextRequestId++;
        _requests[reqId] = RequestData({
            owner: owner,
            isDeposit: false,
            assets: 0,
            shares: shares,
            settledShares: 0,
            refundAssets: 0,
            navRound: navRound,
            totalAssetsSnapshot: totalAssetsSnapshot,
            supplySnapshot: supplySnapshot,
            status: RequestStatus.Executing,
            groupId: bytes32(0)
        });
        requestExpectedValue[reqId] = expectedAssets;
        userEscrowedShares[owner] += shares;
        escrowedShares += shares;
        _userRequestIds[owner].push(reqId);
    }

    function setGroupId(uint256 reqId, bytes32 groupId) external onlyCoreVault {
        _requests[reqId].groupId = groupId;
    }

    ///  ---------------------------Order Settlement---------------------------------

    function validateDepositRequest(bytes32 groupId, uint256 reqId) external view {
        _validateDepositRequest(groupId, reqId);
    }

    function _validateDepositRequest(bytes32 groupId, uint256 reqId) internal view {
        RequestData storage r = _requests[reqId];
        require(r.status == RequestStatus.Executing && r.isDeposit, "RequestManager: bad deposit");
        require(r.groupId == groupId, "RequestManager: wrong group");
    }

    function settleDepositRequest(uint256 reqId, uint256 refundUsdc)
        external
        onlyCoreVault
        returns (DepositSettlement memory result)
    {
        return _settleDepositRequest(reqId, refundUsdc);
    }

    function _settleDepositRequest(uint256 reqId, uint256 refundUsdc)
        internal
        returns (DepositSettlement memory result)
    {
        RequestData storage r = _requests[reqId];
        require(refundUsdc <= r.assets, "RequestManager: refund high");
        uint256 spent = r.assets - refundUsdc;
        uint256 shares = _previewDeposit(FixedPoint.usdcToWad(r.assets), r.totalAssetsSnapshot, r.supplySnapshot);
        r.settledShares = shares;
        r.refundAssets = refundUsdc;
        r.status = RequestStatus.Completed;
        result = DepositSettlement({ owner: r.owner, spent: spent, shares: shares, refund: refundUsdc });
    }

    function validateRedeemRequest(bytes32 groupId, uint256 reqId)
        external
        view
        returns (uint256 totalShares)
    {
        return _validateRedeemRequest(groupId, reqId);
    }

    function _validateRedeemRequest(bytes32 groupId, uint256 reqId)
        internal
        view
        returns (uint256 totalShares)
    {
        RequestData storage r = _requests[reqId];
        require(r.status == RequestStatus.Executing && !r.isDeposit, "RequestManager: bad redeem");
        require(r.groupId == groupId, "RequestManager: wrong group");
        totalShares = r.shares;
    }

    function settleRedeemRequest(
        uint256 reqId,
        uint256 usdcReceived
    ) external onlyCoreVault returns (RedeemSettlement memory result) {
        return _settleRedeemRequest(reqId, usdcReceived);
    }

    function _settleRedeemRequest(
        uint256 reqId,
        uint256 usdcReceived
    ) internal returns (RedeemSettlement memory result) {
        RequestData storage r = _requests[reqId];
        uint256 fulfilledShares = r.shares;
        uint256 returnedShares = r.shares - fulfilledShares;
        r.status = RequestStatus.Completed;
        r.settledShares = r.shares;
        userEscrowedShares[r.owner] -= r.shares;
        escrowedShares -= r.shares;
        result = RedeemSettlement({
            owner: r.owner,
            requestedShares: r.shares,
            fulfilledShares: fulfilledShares,
            returnedShares: returnedShares,
            payout: usdcReceived
        });
    }

    function settleFailure(bytes32 groupId, uint256 reqId, uint256 returnedUsdc)
        external
        onlyCoreVault
        returns (FailureSettlement memory result)
    {
        return _settleFailure(groupId, reqId, returnedUsdc);
    }

    function _settleFailure(bytes32 groupId, uint256 reqId, uint256 returnedUsdc)
        internal
        returns (FailureSettlement memory result)
    {
        RequestData storage r = _requests[reqId];
        require(r.status == RequestStatus.Executing, "RequestManager: bad fail");
        require(r.groupId == groupId, "RequestManager: wrong group");
        r.status = RequestStatus.Failed;
        if (r.isDeposit) {
            require(returnedUsdc <= r.assets, "RequestManager: refund high");
            r.refundAssets = returnedUsdc;
            result = FailureSettlement({ owner: r.owner, isDeposit: true, shares: 0, refund: returnedUsdc });
        } else {
            require(returnedUsdc == 0, "RequestManager: no usdc");
            userEscrowedShares[r.owner] -= r.shares;
            escrowedShares -= r.shares;
            result = FailureSettlement({ owner: r.owner, isDeposit: false, shares: r.shares, refund: 0 });
        }
    }

    ///  ---------------------------Settlement Entry Points---------------------------------

    function settleDeposit(
        bytes32 groupId,
        uint256 reqId,
        uint256 refundUsdc
    ) external onlyCoreVault returns (DepositSettlement memory result) {
        if (settledGroup[groupId]) return result;
        settledGroup[groupId] = true;
        _validateDepositRequest(groupId, reqId);
        result = _settleDepositRequest(reqId, refundUsdc);
    }

    function settleRedeem(
        bytes32 groupId,
        uint256 reqId,
        uint256 usdcReceived
    ) external onlyCoreVault returns (RedeemSettlement memory result) {
        if (settledGroup[groupId]) return result;
        settledGroup[groupId] = true;
        _validateRedeemRequest(groupId, reqId);
        result = _settleRedeemRequest(reqId, usdcReceived);
    }

    function settleRequestFailure(
        bytes32 groupId,
        uint256 reqId,
        uint256 returnedUsdc
    ) external onlyCoreVault returns (FailureSettlement memory result) {
        if (settledGroup[groupId]) return result;
        settledGroup[groupId] = true;
        result = _settleFailure(groupId, reqId, returnedUsdc);
    }

    function isSettled(bytes32 groupId) external view returns (bool) {
        return settledGroup[groupId];
    }


    ///  ---------------------------Public View---------------------------------

    function getRequest(uint256 reqId) external view returns (RequestData memory) {
        return _requests[reqId];
    }

    function userRequestCount(address owner) external view returns (uint256) {
        return _userRequestIds[owner].length;
    }

    function userRequestIdAt(address owner, uint256 index) external view returns (uint256) {
        return _userRequestIds[owner][index];
    }

    function _previewDeposit(uint256 assets18, uint256 totalAssets, uint256 supply) internal pure returns (uint256) {
        if (supply == 0 || totalAssets == 0) return assets18;
        return assets18 * supply / totalAssets;
    }

    function _onlyCoreVault() internal view {
        require(msg.sender == coreVault, "RequestManager: not core");
    }

}


// ===== FILE: src/tokens/ShareToken.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { ERC20 } from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import { IERC165 } from "@openzeppelin/contracts/utils/introspection/IERC165.sol";
import { CoreAccess } from "../base/CoreAccess.sol";
import { IBlocklist } from "../interfaces/IBlocklist.sol";
import { IShareToken } from "../interfaces/IShareToken.sol";
import { Roles } from "../libraries/Roles.sol";

/// @title ShareToken
contract ShareToken is ERC20, CoreAccess, IShareToken {
    IBlocklist public blocklist;
    string private _shareName;
    string private _shareSymbol;
    mapping(address => address) private _vaultByAsset;
    mapping(address => address) private _assetByVault;
    bool private _shareTokenInitialized;

    constructor(string memory name_, string memory symbol_, address admin, address blocklist_)
        ERC20(name_, symbol_)
        CoreAccess(admin)
    {
        if (admin != address(0)) _initializeShareToken(name_, symbol_, blocklist_);
    }

    function initialize(string memory name_, string memory symbol_, address admin, address blocklist_) external {
        _initializeCoreAccess(admin);
        _initializeShareToken(name_, symbol_, blocklist_);
    }

    function _initializeShareToken(string memory name_, string memory symbol_, address blocklist_) internal {
        require(!_shareTokenInitialized, "ShareToken: initialized");
        require(blocklist_ != address(0), "ShareToken: zero blocklist");
        _shareTokenInitialized = true;
        _shareName = name_;
        _shareSymbol = symbol_;
        blocklist = IBlocklist(blocklist_);
    }

    function name() public view override returns (string memory) {
        return _shareName;
    }

    function symbol() public view override returns (string memory) {
        return _shareSymbol;
    }

    function supportsInterface(bytes4 interfaceId) external pure returns (bool) {
        return interfaceId == type(IERC165).interfaceId || interfaceId == type(IShareToken).interfaceId;
    }

    function mint(address to, uint256 amount) external {
        require(isVault(msg.sender) || hasRole(Roles.TOKEN_MINTER, msg.sender), "ShareToken: not minter");
        _mint(to, amount);
    }

    function burn(address from, uint256 amount) external {
        require(isVault(msg.sender) || hasRole(Roles.TOKEN_MINTER, msg.sender), "ShareToken: not burner");
        _burn(from, amount);
    }

    function addVault(address asset_, address vault_) external onlyAdmin {
        require(asset_ != address(0) && vault_ != address(0), "ShareToken: zero address");
        address oldVault = _vaultByAsset[asset_];
        if (oldVault != address(0)) {
            delete _assetByVault[oldVault];
        }
        _vaultByAsset[asset_] = vault_;
        _assetByVault[vault_] = asset_;
        emit VaultAdded(asset_, vault_);
    }

    function removeVault(address asset_) external onlyAdmin {
        address oldVault = _vaultByAsset[asset_];
        require(oldVault != address(0), "ShareToken: missing vault");
        delete _vaultByAsset[asset_];
        delete _assetByVault[oldVault];
        emit VaultRemoved(asset_, oldVault);
    }

    function vault(address asset_) external view returns (address) {
        return _vaultByAsset[asset_];
    }

    function asset(address vault_) external view returns (address) {
        return _assetByVault[vault_];
    }

    function isVault(address vault_) public view returns (bool) {
        return _assetByVault[vault_] != address(0);
    }

    function _transfer(address from, address to, uint256 value) internal override {
        require(!blocklist.isBlocked(from) && !blocklist.isBlocked(to), "ShareToken: blocked");
        super._transfer(from, to, value);
    }

    function _mint(address to, uint256 value) internal override {
        require(!blocklist.isBlocked(to), "ShareToken: blocked");
        super._mint(to, value);
    }
}


// ===== FILE: src/access/VaultAuthority.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { IVaultAuthority } from "../interfaces/IVaultAuthority.sol";
import { Roles } from "../libraries/Roles.sol";

/// @title VaultAuthority
/// @notice Shared authority for one vault deployment. Super admins only manage admins; admins manage vault roles.
contract VaultAuthority is IVaultAuthority {
    mapping(bytes32 => mapping(address => bool)) private _roles;
    bool private _initialized;

    modifier onlySuperAdmin() {
        _onlySuperAdmin();
        _;
    }

    modifier onlyAdmin() {
        _onlyAdmin();
        _;
    }

    constructor(address superAdmin, address admin) {
        if (superAdmin != address(0)) {
            _initialize(superAdmin, admin);
        }
    }

    function initialize(address superAdmin, address admin) external {
        _initialize(superAdmin, admin);
    }

    function _initialize(address superAdmin, address admin) internal {
        require(!_initialized, "VaultAuthority: initialized");
        require(superAdmin != address(0) && admin != address(0), "VaultAuthority: zero address");
        _initialized = true;
        _grantRole(Roles.SUPER_ADMIN_ROLE, superAdmin);
        _grantRole(Roles.DEFAULT_ADMIN_ROLE, admin);
        if (msg.sender != superAdmin && msg.sender != admin) _grantRole(Roles.DEFAULT_ADMIN_ROLE, msg.sender);
    }

    function hasRole(bytes32 role, address account) external view returns (bool) {
        return _roles[role][account];
    }

    function grantAdmin(address account) external onlySuperAdmin {
        require(account != address(0), "VaultAuthority: zero account");
        _grantRole(Roles.DEFAULT_ADMIN_ROLE, account);
    }

    function revokeAdmin(address account) external onlySuperAdmin {
        _revokeRole(Roles.DEFAULT_ADMIN_ROLE, account);
    }

    function grantRole(bytes32 role, address account) external {
        require(account != address(0), "VaultAuthority: zero account");
        if (role == Roles.DEFAULT_ADMIN_ROLE) {
            require(_roles[Roles.SUPER_ADMIN_ROLE][msg.sender], "VaultAuthority: not super admin");
        } else {
            require(_roles[Roles.DEFAULT_ADMIN_ROLE][msg.sender], "VaultAuthority: not admin");
            require(role != Roles.SUPER_ADMIN_ROLE, "VaultAuthority: super admin fixed");
        }
        _grantRole(role, account);
    }

    function revokeRole(bytes32 role, address account) external {
        if (role == Roles.DEFAULT_ADMIN_ROLE) {
            require(_roles[Roles.SUPER_ADMIN_ROLE][msg.sender], "VaultAuthority: not super admin");
        } else {
            require(_roles[Roles.DEFAULT_ADMIN_ROLE][msg.sender], "VaultAuthority: not admin");
            require(role != Roles.SUPER_ADMIN_ROLE, "VaultAuthority: super admin fixed");
        }
        _revokeRole(role, account);
    }

    function renounceRole(bytes32 role, address account) external {
        require(account == msg.sender, "VaultAuthority: can only renounce self");
        require(role != Roles.SUPER_ADMIN_ROLE, "VaultAuthority: super admin fixed");
        _revokeRole(role, account);
    }

    function _grantRole(bytes32 role, address account) internal {
        if (_roles[role][account]) return;
        _roles[role][account] = true;
        emit RoleGranted(role, account, msg.sender);
    }

    function _revokeRole(bytes32 role, address account) internal {
        if (!_roles[role][account]) return;
        _roles[role][account] = false;
        emit RoleRevoked(role, account, msg.sender);
    }

    function _onlySuperAdmin() internal view {
        require(_roles[Roles.SUPER_ADMIN_ROLE][msg.sender], "VaultAuthority: not super admin");
    }

    function _onlyAdmin() internal view {
        require(_roles[Roles.DEFAULT_ADMIN_ROLE][msg.sender], "VaultAuthority: not admin");
    }
}


// ===== FILE: src/access/Whitelist.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { CoreAccess } from "../base/CoreAccess.sol";
import { IWhitelist } from "../interfaces/IWhitelist.sol";

/// @title Whitelist
contract Whitelist is CoreAccess, IWhitelist {
    mapping(address => bool) private _users;
    mapping(address => bool) private _assets;
    mapping(address => uint8) private _assetDecimals;

    event UserWhitelistSet(address indexed account, bool allowed);
    event AssetWhitelistSet(address indexed asset, bool allowed);
    event AssetDecimalsSet(address indexed asset, uint8 decimals);

    constructor(address admin) CoreAccess(admin) { }

    function initialize(address admin) external {
        _initializeCoreAccess(admin);
    }

    function setUser(address account, bool allowed) external onlyAdmin {
        _users[account] = allowed;
        emit UserWhitelistSet(account, allowed);
    }

    function setUsers(address[] calldata accounts, bool[] calldata allowed)
        external
        onlyAdmin
    {
        require(accounts.length == allowed.length, "Whitelist: length mismatch");
        for (uint256 i; i < accounts.length; ++i) {
            _users[accounts[i]] = allowed[i];
            emit UserWhitelistSet(accounts[i], allowed[i]);
        }
    }

    function setAsset(address asset, bool allowed) external onlyAdmin {
        _assets[asset] = allowed;
        emit AssetWhitelistSet(asset, allowed);
    }

    function setAssetWithMetadata(address asset, bool allowed, uint8 decimals)
        external
        onlyAdmin
    {
        _assets[asset] = allowed;
        _assetDecimals[asset] = decimals;
        emit AssetWhitelistSet(asset, allowed);
        emit AssetDecimalsSet(asset, decimals);
    }

    function setAssets(address[] calldata assets, bool[] calldata allowed) external onlyAdmin {
        require(assets.length == allowed.length, "Whitelist: length mismatch");
        for (uint256 i; i < assets.length; ++i) {
            _assets[assets[i]] = allowed[i];
            emit AssetWhitelistSet(assets[i], allowed[i]);
        }
    }

    function isWhitelisted(address account) external view returns (bool) {
        return _users[account];
    }

    function isAssetAllowed(address asset) external view returns (bool) {
        return _assets[asset];
    }

    function assetDecimals(address asset) external view returns (uint8) {
        return _assetDecimals[asset];
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


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/proxy/ERC1967/ERC1967Proxy.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.6.0) (proxy/ERC1967/ERC1967Proxy.sol)

pragma solidity ^0.8.22;

import {Proxy} from "../Proxy.sol";
import {ERC1967Utils} from "./ERC1967Utils.sol";

/**
 * @dev This contract implements an upgradeable proxy. It is upgradeable because calls are delegated to an
 * implementation address that can be changed. This address is stored in storage in the location specified by
 * https://eips.ethereum.org/EIPS/eip-1967[ERC-1967], so that it doesn't conflict with the storage layout of the
 * implementation behind the proxy.
 */
contract ERC1967Proxy is Proxy {
    /**
     * @dev The proxy is left uninitialized.
     */
    error ERC1967ProxyUninitialized();

    /**
     * @dev Initializes the upgradeable proxy with an initial implementation specified by `implementation`.
     *
     * Provided `_data` is passed in a delegate call to `implementation`. This will typically be an encoded function
     * call, and allows initializing the storage of the proxy like a Solidity constructor. By default construction
     * will fail if `_data` is empty. This behavior can be overridden using a custom {_unsafeAllowUninitialized} that
     * returns true. In that case, empty `_data` is ignored and no delegate call to the implementation is performed
     * during construction.
     *
     * Requirements:
     *
     * - If `data` is empty, `msg.value` must be zero.
     */
    constructor(address implementation, bytes memory _data) payable {
        if (!_unsafeAllowUninitialized() && _data.length == 0) {
            revert ERC1967ProxyUninitialized();
        }
        ERC1967Utils.upgradeToAndCall(implementation, _data);
    }

    /**
     * @dev Returns the current implementation address.
     *
     * TIP: To get this value clients can read directly from the storage slot shown below (specified by ERC-1967) using
     * the https://eth.wiki/json-rpc/API#eth_getstorageat[`eth_getStorageAt`] RPC call.
     * `0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc`
     */
    function _implementation() internal view virtual override returns (address) {
        return ERC1967Utils.getImplementation();
    }

    /**
     * @dev Returns whether the proxy can be left uninitialized.
     *
     * NOTE: Override this function to allow the proxy to be left uninitialized.
     * Consider uninitialized proxies might be susceptible to man-in-the-middle threats
     * where the proxy is replaced with a malicious one.
     */
    function _unsafeAllowUninitialized() internal pure virtual returns (bool) {
        return false;
    }
}


// ===== FILE: src/modules/AssetCustodian.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { IERC20 } from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import { IERC20Metadata } from "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import { SafeERC20 } from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import { Pausable } from "@openzeppelin/contracts/utils/Pausable.sol";
import { ReentrancyGuard } from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import { CoreAccess } from "../base/CoreAccess.sol";
import { Roles } from "../libraries/Roles.sol";
import { IACCallback } from "../interfaces/IACCallback.sol";
import { IAssetCustodian } from "../interfaces/IAssetCustodian.sol";
import { INAVOracle } from "../interfaces/INAVOracle.sol";
import { IOndoProxy } from "../interfaces/IOndoProxy.sol";
import { IVaultSettlement } from "../interfaces/IVaultSettlement.sol";
import { BatchHint, BatchStatus, RebalanceBuy, RebalanceTrigger, Trade } from "../interfaces/Types.sol";

interface ICoreVaultAllocation {
    function getTargetAllocation() external view returns (address[] memory tokens, uint256[] memory weights);
}

/// @title AssetCustodian
/// @notice Execution hub between CoreVault and external proxy: temporary fund/asset storage, routing policies, receiving proxy
contract AssetCustodian is
    CoreAccess,
    Pausable,
    ReentrancyGuard,
    IAssetCustodian,
    IACCallback
{
    using SafeERC20 for IERC20;

    enum Operation {
        Deposit,
        Redeem,
        Rebalance
    }

    struct GroupState {
        Operation op;
        BatchStatus status;
        uint256 reqId;
        uint256 inAmount;
        bytes32 rbId;
        uint256 traceCount;
        uint256 settledTraceCount;
        bool hasFailure;
        uint256 depositProxyTraceCount;
        uint256 depositProxySuccessCount;
        uint256 localDepositUsdcAmount;
        uint256 returnedUsdc;
        uint256 navRound;
        bytes routeData;
        bool dispatching;

        RebalanceBuy[] buyPlans;
        uint256 rebalanceUsdcProceeds;
        bool rebalanceBuyDispatched;
        BatchHint rebalanceHint;

        address[] settledTokens;
        uint256[] settledAmounts;
        bytes failReason;
        RebalanceTrigger rebalanceTrigger;
    }

    struct TraceState {
        bytes32 groupId;
        Operation op;
        address inputToken;
        address outputToken;
        uint256 minAmountOut;
        bool settled;
    }

    address public coreVault;
    IOndoProxy public ondoProxy;
    IERC20 public usdc;
    INAVOracle public navOracle;
    uint256 public nonce;
    bool private _assetCustodianInitialized;

    mapping(bytes32 => GroupState) private _groups;
    mapping(uint256 => bytes32) public requestToGroup;
    mapping(bytes32 => TraceState) public traceState;

    event GroupCreated(bytes32 indexed groupId, Operation indexed op, uint256 requestCount);
    event TraceDispatched(
        bytes32 indexed groupId,
        bytes32 indexed traceId,
        Operation indexed op,
        address inputToken,
        address outputToken,
        uint256 amount
    );
    event TraceSettled(bytes32 indexed groupId, bytes32 indexed traceId, bool success, address token, uint256 amount);
    event ProxyCallbackFailed(bytes32 indexed groupId, bytes32 indexed traceId, Operation indexed op, bytes data);
    event GroupSettled(bytes32 indexed groupId, Operation indexed op);
    event GroupFailed(bytes32 indexed groupId, bytes reason);
    event RebalanceSnapshot(
        bytes32 indexed rbId,
        bytes32 indexed groupId,
        RebalanceTrigger indexed trigger,
        bool afterRebalance,
        bytes32 routeHash,
        bytes32 allocationHash,
        bytes routeData,
        address[] tokens,
        uint256[] amounts,
        uint256[] weights
    );
    event RebalanceLateCallbackIgnored(bytes32 indexed groupId, bytes32 indexed rbId);
    event OndoProxySet(address indexed proxy);

    modifier onlyCoreVault() {
        _onlyCoreVault();
        _;
    }

    modifier onlyOndoProxy() {
        _onlyOndoProxy();
        _;
    }

    constructor(
        address admin,
        address coreVault_,
        address ondoProxy_,
        address usdc_,
        address navOracle_
    ) CoreAccess(admin) {
        if (admin != address(0)) {
            _initializeAssetCustodian(coreVault_, ondoProxy_, usdc_, navOracle_);
        }
    }

    function initialize(
        address admin,
        address coreVault_,
        address ondoProxy_,
        address usdc_,
        address navOracle_
    ) external {
        _initializeCoreAccess(admin);
        _initializeAssetCustodian(coreVault_, ondoProxy_, usdc_, navOracle_);
    }

    function _initializeAssetCustodian(
        address coreVault_,
        address ondoProxy_,
        address usdc_,
        address navOracle_
    ) internal {
        require(!_assetCustodianInitialized, "AC:INIT");
        require(
            coreVault_ != address(0) && ondoProxy_ != address(0) && usdc_ != address(0) && navOracle_ != address(0),
            "AC: zero address"
        );
        _assetCustodianInitialized = true;
        coreVault = coreVault_;
        ondoProxy = IOndoProxy(ondoProxy_);
        usdc = IERC20(usdc_);
        navOracle = INAVOracle(navOracle_);
    }

    function setOndoProxy(address proxy) external onlyAdmin {
        require(proxy != address(0), "AC:ZP");
        ondoProxy = IOndoProxy(proxy);
        emit OndoProxySet(proxy);
    }

    function pause() external onlyRole(Roles.PAUSE_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(Roles.PAUSE_ROLE) {
        _unpause();
    }

    function groupState(bytes32 groupId) external view returns (Operation op, BatchStatus status, uint256 inAmount) {
        GroupState storage b = _groups[groupId];
        return (b.op, b.status, b.inAmount);
    }

    function groupReqId(bytes32 groupId) external view returns (uint256) {
        return _groups[groupId].reqId;
    }

    function groupTraceIds(bytes32 groupId) external view returns (bytes32[] memory) {
        GroupState storage b = _groups[groupId];
        bytes32[] memory ids = new bytes32[](b.traceCount);
        for (uint256 i; i < b.traceCount; ++i) {
            ids[i] = _traceId(groupId, i);
        }
        return ids;
    }

    function previewNextGroupId() external view returns (bytes32) {
        return _efficientHash(address(this), block.chainid, nonce + 1);
    }

    ///  ---------------------------Execution Function---------------------------------

    function executeDeposit(uint256 reqId, uint256 totalUsdc, BatchHint calldata hint)
        external
        onlyCoreVault
        whenNotPaused
        nonReentrant
        returns (bytes32 groupId)
    {
        require(reqId != 0 && totalUsdc > 0, "AC:ED");
        groupId = _newGroupId();
        GroupState storage b = _groups[groupId];
        b.op = Operation.Deposit;
        b.status = BatchStatus.PendingProxy;
        b.inAmount = totalUsdc;
        b.navRound = navOracle.latestRound();
        b.reqId = reqId;
        requestToGroup[reqId] = groupId;
        (address[] memory targetTokens, uint256[] memory bps) = ICoreVaultAllocation(coreVault).getTargetAllocation();
        require(targetTokens.length > 0 && targetTokens.length == bps.length, "AC:TA");
        uint256 totalBps;
        for (uint256 i; i < bps.length; ++i) {
            totalBps += bps[i];
        }
        require(totalBps == 10_000, "AC:BW");
        uint256 dispatched;
        b.dispatching = true;
        for (uint256 i; i < targetTokens.length; ++i) {
            uint256 orderAmount = i == targetTokens.length - 1 ? totalUsdc - dispatched : totalUsdc * bps[i] / 10_000;
            dispatched += orderAmount;
            if (orderAmount == 0) continue;
            _dispatchProxyOrder(groupId, Operation.Deposit, address(usdc), targetTokens[i], orderAmount, 0, hint);
        }
        require(b.traceCount > 0, "AC:NT");
        b.dispatching = false;
        emit GroupCreated(groupId, Operation.Deposit, 1);
        _finalizeIfReady(groupId, b);
    }

    function executeRedeem(
        uint256 reqId,
        address[] calldata etfTokens,
        uint256[] calldata amounts,
        BatchHint calldata hint
    ) external onlyCoreVault whenNotPaused nonReentrant returns (bytes32 groupId) {
        require(reqId != 0, "AC:ER");
        require(etfTokens.length == amounts.length, "AC:LEN");
        groupId = _newGroupId();
        GroupState storage b = _groups[groupId];
        b.op = Operation.Redeem;
        b.status = BatchStatus.PendingProxy;
        b.navRound = navOracle.latestRound();
        b.reqId = reqId;
        requestToGroup[reqId] = groupId;
        b.dispatching = true;
        for (uint256 i; i < etfTokens.length; ++i) {
            if (amounts[i] == 0) continue;
            _dispatchProxyOrder(groupId, Operation.Redeem, etfTokens[i], address(usdc), amounts[i], 0, hint);
        }
        require(b.traceCount > 0, "AC:NT");
        b.dispatching = false;
        emit GroupCreated(groupId, Operation.Redeem, 1);
        _finalizeIfReady(groupId, b);
    }

    function executeRebalance(
        bytes32 rbId,
        Trade[] calldata sells,
        RebalanceBuy[] calldata buys,
        RebalanceTrigger trigger,
        BatchHint calldata hint
    ) external onlyCoreVault whenNotPaused nonReentrant returns (bytes32 groupId) {
        groupId = _newGroupId();
        GroupState storage b = _groups[groupId];
        b.op = Operation.Rebalance;
        b.status = BatchStatus.PendingProxy;
        b.rbId = rbId;
        b.rebalanceHint = hint;
        b.navRound = navOracle.latestRound();
        b.rebalanceTrigger = trigger;
        _copyRebalanceBuyPlans(b, buys);
        b.routeData = abi.encode(sells, buys);
        _emitRebalanceSnapshot(groupId, rbId, trigger, false, b.routeData);
        b.dispatching = true;
        for (uint256 i; i < sells.length; ++i) {
            if (sells[i].amountIn == 0) continue;
            require(sells[i].tokenOut == address(usdc), "AC:SU");
            _dispatchProxyOrder(
                groupId,
                Operation.Rebalance,
                sells[i].tokenIn,
                address(usdc),
                sells[i].amountIn,
                sells[i].minAmountOut,
                hint
            );
        }
        require(b.traceCount > 0, "AC:NT");
        b.dispatching = false;
        emit GroupCreated(groupId, Operation.Rebalance, 0);
        _finalizeIfReady(groupId, b);
    }

    function _dispatchProxyOrder(
        bytes32 groupId,
        Operation op,
        address inputToken,
        address outputToken,
        uint256 amount,
        uint256 minAmountOut,
        BatchHint memory hint
    ) internal {
        uint256 traceIndex = _groups[groupId].traceCount;
        bytes32 traceId = _traceId(groupId, traceIndex);
        traceState[traceId] = TraceState({
            groupId: groupId,
            op: op,
            inputToken: inputToken,
            outputToken: outputToken,
            minAmountOut: minAmountOut,
            settled: false
        });
        _groups[groupId].traceCount += 1;
        if (inputToken == outputToken) {
            // Intercept sub-orders locally
            emit TraceDispatched(groupId, traceId, op, inputToken, outputToken, amount);
            _onLocalTraceSettled(traceId, outputToken, amount);
            return;
        }
        if (op == Operation.Deposit) {
            _groups[groupId].depositProxyTraceCount += 1;
        }

        bytes memory extraData = abi.encode(inputToken, outputToken, amount);
        IOndoProxy.DepositIntent memory intent = IOndoProxy.DepositIntent({
            receiver: address(this),
            outputToken: outputToken,
            destinationChainId: block.chainid,
            maxSlippageBps: hint.maxSlippageBps,
            maxLossPercentBps: hint.maxLossPercentBps,
            extraData: extraData
        });
        IERC20(inputToken).forceApprove(address(ondoProxy), amount);
        ondoProxy.depositERC20(inputToken, amount, traceId, intent);
        IERC20(inputToken).forceApprove(address(ondoProxy), 0);

        emit TraceDispatched(groupId, traceId, op, inputToken, outputToken, amount);
    }

    function _onLocalTraceSettled(bytes32 traceId, address outputToken, uint256 amount) internal {
        TraceState storage t = traceState[traceId];
        GroupState storage b = _groups[t.groupId];
        t.settled = true;
        b.settledTraceCount += 1;
        uint256 actual = _recordSuccessfulTrace(t.groupId, b, t, outputToken, amount);
        if (t.op == Operation.Deposit && outputToken == address(usdc)) {
            b.localDepositUsdcAmount += actual;
        }
        emit TraceSettled(t.groupId, traceId, true, outputToken, amount);
        _finalizeIfReady(t.groupId, b);
    }


    ///  ---------------------------Callback Function---------------------------------

    /// @notice Proxy Single trace order settlement callback. AC aggregates the results of all traces within the group, and notifies CoreVault for final settlement only after all callbacks are completed.
    function onOrderSettled(bytes32 traceId, bool success, address token, uint256 amount, bytes calldata data)
        external
        onlyOndoProxy
        nonReentrant
    {
        TraceState storage t = traceState[traceId];
        bytes32 groupId = t.groupId;
        require(groupId != bytes32(0), "AC:UT");
        require(!t.settled, "AC:TS");
        GroupState storage b = _groups[groupId];

        if (b.status == BatchStatus.Settled || b.status == BatchStatus.Failed) {
            _forwardLateAssetToCore(token, amount);
            if (b.op == Operation.Rebalance) emit RebalanceLateCallbackIgnored(groupId, b.rbId);
            return;
        }

        require(b.status == BatchStatus.PendingProxy && b.op == t.op, "AC:BTC");
        t.settled = true;
        b.settledTraceCount += 1;
        if (success) {
            uint256 actual = _recordSuccessfulTrace(groupId, b, t, token, amount);
            if (t.op == Operation.Deposit && actual > 0) {
                b.depositProxySuccessCount += 1;
            }
        } else {
            emit ProxyCallbackFailed(groupId, traceId, t.op, data);
            if (data.length != 0 && b.failReason.length == 0) b.failReason = data;
            _recordFailedTrace(groupId, b, t, token, amount);
        }

        emit TraceSettled(groupId, traceId, success, token, amount);

        _finalizeIfReady(groupId, b);
    }

    function _finalizeIfReady(bytes32 groupId, GroupState storage b) internal {
        if (b.dispatching || b.settledTraceCount != b.traceCount) return;
        if (b.op == Operation.Rebalance && !b.rebalanceBuyDispatched) {
            _finalizeRebalanceSellPhase(groupId, b);
        } else {
            _finalizeGroup(groupId, b);
        }
    }

    function _recordSuccessfulTrace(
        bytes32 groupId,
        GroupState storage b,
        TraceState storage t,
        address token,
        uint256 amount
    ) internal returns (uint256 actual) {
        if (t.minAmountOut > 0) require(amount >= t.minAmountOut, "AC:UF");
        if (t.op == Operation.Deposit) {
            require(token == t.outputToken, "AC:BDT");
            actual = amount;
            _appendSettledToken(b, token, actual);
        } else if (t.op == Operation.Redeem) {
            require(token == address(usdc), "AC:BRT");
            b.returnedUsdc += amount;
            actual = amount;
        } else if (t.op == Operation.Rebalance) {
            require(token == t.outputToken, "AC:BBT");
            if (!b.rebalanceBuyDispatched) {
                require(token == address(usdc), "AC:BSO");
                b.rebalanceUsdcProceeds += amount;
                actual = amount;
            } else {
                actual = _safeTransferToCoreByBalanceDiff(token, amount);
                _appendSettledToken(b, token, actual);
            }
        }
    }

    function _recordFailedTrace(
        bytes32 groupId,
        GroupState storage b,
        TraceState storage t,
        address token,
        uint256 amount
    ) internal {
        b.hasFailure = true;
        if (b.failReason.length == 0) b.failReason = bytes("AC: proxy order failed");
        if (t.op == Operation.Deposit) {
            require(token == address(usdc), "AC:BDR");
            b.returnedUsdc += amount;
        } else if (t.op == Operation.Redeem) {
            require(token == t.inputToken, "AC:BRR");
            _returnSingleUnsoldToken(token, amount);
        } else if (t.op == Operation.Rebalance) {
            require(token == t.inputToken || token == t.outputToken, "AC:BRF");
            if (token == address(usdc)) {
                b.rebalanceUsdcProceeds += amount;
            } else {
                _forwardLateAssetToCore(token, amount);
            }
        }
    }

    function _finalizeRebalanceSellPhase(bytes32 groupId, GroupState storage b) internal {
        if (b.hasFailure) {
            _flushRebalanceUsdcToCore(b);
            b.status = BatchStatus.Failed;
            IVaultSettlement(coreVault).settleRebalanceFailure(b.rbId, b.failReason);
            emit GroupFailed(groupId, b.failReason);
            return;
        }
        if (b.rebalanceUsdcProceeds == 0) {
            b.status = BatchStatus.Failed;
            b.failReason = bytes("AC: no sell proceeds");
            IVaultSettlement(coreVault).settleRebalanceFailure(b.rbId, b.failReason);
            emit GroupFailed(groupId, b.failReason);
            return;
        }
        b.rebalanceBuyDispatched = true;
        uint256 beforeTraceCount = b.traceCount;
        uint256 totalProceeds = b.rebalanceUsdcProceeds;
        BatchHint memory hint = _rebalanceHint(b);
        uint256 totalRequestedAmountIn;
        for (uint256 i; i < b.buyPlans.length; ++i) {
            require(b.buyPlans[i].amountIn > 0, "AC:ZBA");
            totalRequestedAmountIn += b.buyPlans[i].amountIn;
        }
        require(totalRequestedAmountIn > 0, "AC:ZTA");
        uint256 dispatchedUsdc;
        b.dispatching = true;
        for (uint256 i; i < b.buyPlans.length; ++i) {
            uint256 orderAmount;
            if (i < b.buyPlans.length - 1) {
                orderAmount = totalProceeds * b.buyPlans[i].amountIn / totalRequestedAmountIn;
            } else {
                orderAmount = totalProceeds - dispatchedUsdc;
            }
            require(orderAmount > 0, "AC:ZBA");
            dispatchedUsdc += orderAmount;
            b.rebalanceUsdcProceeds -= orderAmount;
            _dispatchProxyOrder(
                groupId,
                Operation.Rebalance,
                b.buyPlans[i].tokenIn,
                b.buyPlans[i].tokenOut,
                orderAmount,
                b.buyPlans[i].minAmountOut,
                hint
            );
        }
        b.dispatching = false;
        require(b.traceCount > beforeTraceCount, "AC:NBT");
        _finalizeIfReady(groupId, b);
    }

    function _finalizeGroup(bytes32 groupId, GroupState storage b)
        internal
    {
        if (b.op == Operation.Deposit) {
            _finalizeDepositGroup(groupId, b);
        } else if (b.op == Operation.Redeem) {
            _finalizeRedeemGroup(groupId, b);
        } else if (b.op == Operation.Rebalance) {
            _finalizeRebalanceGroup(groupId, b);
        }
    }

    function _finalizeDepositGroup(bytes32 groupId, GroupState storage b) internal {
        if (b.returnedUsdc > 0) usdc.safeTransfer(coreVault, b.returnedUsdc);
        if (b.depositProxyTraceCount > 0 && b.depositProxySuccessCount == 0) {
            _transferSettledAssetsToCore(b);
            uint256 refundUsdc = b.returnedUsdc + b.localDepositUsdcAmount;
            b.status = BatchStatus.Failed;
            if (b.failReason.length == 0) b.failReason = bytes("AC: no proxy deposit fills");
            IVaultSettlement(coreVault).settleFailure(groupId, b.reqId, refundUsdc, b.failReason);
            emit GroupFailed(groupId, b.failReason);
            return;
        }
        if (b.settledTokens.length == 0) {
            b.status = BatchStatus.Failed;
            IVaultSettlement(coreVault).settleFailure(groupId, b.reqId, b.returnedUsdc, b.failReason);
            emit GroupFailed(groupId, b.failReason);
            return;
        }
        require(b.navRound != 0, "AC:MNR");
        _transferSettledAssetsToCore(b);
        b.status = BatchStatus.Settled;
        IVaultSettlement(coreVault)
            .settleDeposit(groupId, b.reqId, b.settledTokens, b.settledAmounts, b.returnedUsdc, b.navRound);
        emit GroupSettled(groupId, Operation.Deposit);
    }

    function _finalizeRedeemGroup(bytes32 groupId, GroupState storage b) internal {
        if (b.returnedUsdc > 0) usdc.safeTransfer(coreVault, b.returnedUsdc);
        if (b.hasFailure || b.returnedUsdc == 0) {
            b.status = BatchStatus.Failed;
            IVaultSettlement(coreVault).settleFailure(groupId, b.reqId, 0, b.failReason);
            emit GroupFailed(groupId, b.failReason);
            return;
        }
        require(b.navRound != 0, "AC:MNR");
        b.status = BatchStatus.Settled;
        IVaultSettlement(coreVault).settleRedeem(groupId, b.reqId, b.returnedUsdc, b.navRound);
        emit GroupSettled(groupId, Operation.Redeem);
    }

    function _finalizeRebalanceGroup(
        bytes32 groupId,
        GroupState storage b
    ) internal {
        if (b.hasFailure) {
            _flushRebalanceUsdcToCore(b);
            b.status = BatchStatus.Failed;
            IVaultSettlement(coreVault).settleRebalanceFailure(b.rbId, b.failReason);
            emit GroupFailed(groupId, b.failReason);
            return;
        }
        require(b.rebalanceBuyDispatched, "AC:BND");
        _flushRebalanceUsdcToCore(b);
        b.status = BatchStatus.Settled;
        IVaultSettlement(coreVault)
            .settleRebalance(b.rbId, b.settledTokens, b.settledAmounts);
        _emitRebalanceSnapshot(groupId, b.rbId, b.rebalanceTrigger, true, b.routeData);
        emit GroupSettled(groupId, Operation.Rebalance);
    }



    function _appendSettledToken(GroupState storage b, address token, uint256 amount) internal {
        if (amount == 0) return;
        b.settledTokens.push(token);
        b.settledAmounts.push(amount);
    }

    function _transferSettledAssetsToCore(GroupState storage b) internal {
        for (uint256 i; i < b.settledTokens.length; ++i) {
            uint256 amount = b.settledAmounts[i];
            if (amount == 0) continue;
            IERC20(b.settledTokens[i]).safeTransfer(coreVault, amount);
        }
    }

    function _safeTransferToCoreByBalanceDiff(address token, uint256 amount) internal returns (uint256 actual) {
        if (amount == 0) return 0;
        uint256 beforeBalance = IERC20(token).balanceOf(coreVault);
        IERC20(token).safeTransfer(coreVault, amount);
        actual = IERC20(token).balanceOf(coreVault) - beforeBalance;
    }

    function _forwardLateAssetToCore(address token, uint256 amount) internal {
        if (amount > 0) IERC20(token).safeTransfer(coreVault, amount);
    }

    function _returnSingleUnsoldToken(address token, uint256 amount) internal {
        if (amount == 0) return;
        IERC20(token).safeTransfer(coreVault, amount);
    }

    function _newGroupId() internal returns (bytes32) {
        uint256 nextNonce = ++nonce;
        return _efficientHash(address(this), block.chainid, nextNonce);
    }

    function _traceId(bytes32 groupId, uint256 traceIndex) internal view returns (bytes32) {
        return _efficientHash(address(this), block.chainid, groupId, traceIndex);
    }

    function _onlyCoreVault() internal view {
        require(msg.sender == coreVault, "AC:NC");
    }

    function _onlyOndoProxy() internal view {
        require(msg.sender == address(ondoProxy), "AC:NP");
    }

    function _efficientHash(address account, uint256 chainId, uint256 value) internal pure returns (bytes32 result) {
        assembly ("memory-safe") {
            let ptr := mload(0x40)
            mstore(ptr, and(account, 0xffffffffffffffffffffffffffffffffffffffff))
            mstore(add(ptr, 0x20), chainId)
            mstore(add(ptr, 0x40), value)
            result := keccak256(ptr, 0x60)
        }
    }

    function _efficientHash(
        address account,
        uint256 chainId,
        bytes32 groupId,
        uint256 traceIndex
    ) internal pure returns (bytes32 result) {
        assembly ("memory-safe") {
            let ptr := mload(0x40)
            mstore(ptr, and(account, 0xffffffffffffffffffffffffffffffffffffffff))
            mstore(add(ptr, 0x20), chainId)
            mstore(add(ptr, 0x40), groupId)
            mstore(add(ptr, 0x60), traceIndex)
            result := keccak256(ptr, 0x80)
        }
    }

    function _copyRebalanceBuyPlans(GroupState storage b, RebalanceBuy[] calldata buys) internal {
        require(buys.length > 0, "AC:EB");
        for (uint256 i; i < buys.length; ++i) {
            require(buys[i].tokenIn != address(0), "AC:ZBI");
            require(buys[i].tokenOut != address(0), "AC:ZBT");
            require(buys[i].tokenIn == address(usdc), "AC:BUI");
            require(buys[i].amountIn > 0, "AC:ZBA");
            b.buyPlans.push(buys[i]);
        }
    }

    function _rebalanceHint(GroupState storage b) internal view returns (BatchHint memory) {
        return b.rebalanceHint;
    }

    function _emitRebalanceSnapshot(
        bytes32 groupId,
        bytes32 rbId,
        RebalanceTrigger trigger,
        bool afterRebalance,
        bytes memory routeData
    ) internal {
        (
            address[] memory tokens,
            uint256[] memory amounts,
            uint256[] memory weights,
            bytes32 allocationHash
        ) = _rebalanceSnapshotData();
        emit RebalanceSnapshot({
            rbId: rbId,
            groupId: groupId,
            trigger: trigger,
            afterRebalance: afterRebalance,
            routeHash: keccak256(routeData),
            allocationHash: allocationHash,
            routeData: routeData,
            tokens: tokens,
            amounts: amounts,
            weights: weights
        });
    }

    function _rebalanceSnapshotData()
        internal
        view
        returns (address[] memory tokens, uint256[] memory amounts, uint256[] memory weights, bytes32 allocationHash)
    {
        (tokens, weights) = ICoreVaultAllocation(coreVault).getTargetAllocation();
        amounts = new uint256[](tokens.length);
        for (uint256 i; i < tokens.length; ++i) {
            IERC20 token = IERC20(tokens[i]);
            amounts[i] = token.balanceOf(coreVault) + token.balanceOf(address(this));
        }
        allocationHash = keccak256(abi.encode(tokens, amounts, weights));
    }

    function _flushRebalanceUsdcToCore(GroupState storage b) internal {
        uint256 bal = usdc.balanceOf(address(this));
        if (bal == 0) return;
        uint256 amount = bal < b.rebalanceUsdcProceeds ? bal : b.rebalanceUsdcProceeds;
        if (amount > 0) {
            b.rebalanceUsdcProceeds -= amount;
            usdc.safeTransfer(coreVault, amount);
        }
    }

    function _assetUnit(address token) internal view returns (uint256) {
        uint8 decimals = IERC20Metadata(token).decimals();
        require(decimals <= 36, "AC:DEC");
        return 10 ** uint256(decimals);
    }
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


// ===== FILE: src/interfaces/IBlocklist.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IBlocklist {
    function isBlocked(address account) external view returns (bool);
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


// ===== FILE: lib/openzeppelin-contracts/contracts/token/ERC20/utils/SafeERC20.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "../IERC20.sol";

library SafeERC20 {
    function safeTransfer(IERC20 token, address to, uint256 value) internal {
        _call(token, abi.encodeWithSelector(token.transfer.selector, to, value));
    }

    function safeTransferFrom(IERC20 token, address from, address to, uint256 value) internal {
        _call(token, abi.encodeWithSelector(token.transferFrom.selector, from, to, value));
    }

    function safeApprove(IERC20 token, address spender, uint256 value) internal {
        _call(token, abi.encodeWithSelector(token.approve.selector, spender, value));
    }

    function forceApprove(IERC20 token, address spender, uint256 value) internal {
        safeApprove(token, spender, 0);
        safeApprove(token, spender, value);
    }

    function _call(IERC20 token, bytes memory data) private {
        (bool ok, bytes memory ret) = address(token).call(data);
        require(ok && (ret.length == 0 || abi.decode(ret, (bool))), "SafeERC20: failed");
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/Pausable.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

abstract contract Pausable {
    bool private _paused;

    event Paused(address account);
    event Unpaused(address account);

    modifier whenNotPaused() {
        require(!_paused, "Pausable: paused");
        _;
    }

    modifier whenPaused() {
        require(_paused, "Pausable: not paused");
        _;
    }

    function paused() public view virtual returns (bool) {
        return _paused;
    }

    function _pause() internal virtual whenNotPaused {
        _paused = true;
        emit Paused(msg.sender);
    }

    function _unpause() internal virtual whenPaused {
        _paused = false;
        emit Unpaused(msg.sender);
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/ReentrancyGuard.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

abstract contract ReentrancyGuard {
    uint256 private constant NOT_ENTERED = 1;
    uint256 private constant ENTERED = 2;
    uint256 private _status = NOT_ENTERED;

    modifier nonReentrant() {
        require(_status != ENTERED, "ReentrancyGuard: reentrant call");
        _status = ENTERED;
        _;
        _status = NOT_ENTERED;
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/introspection/IERC165.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC165 {
    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}


// ===== FILE: src/interfaces/IAssetCustodian.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { BatchHint, RebalanceBuy, RebalanceTrigger, Trade } from "./Types.sol";

interface IAssetCustodian {
    function previewNextGroupId() external view returns (bytes32);

    function executeDeposit(uint256 reqId, uint256 totalUsdc, BatchHint calldata hint)
        external
        returns (bytes32 groupId);

    function executeRedeem(
        uint256 reqId,
        address[] calldata etfTokens,
        uint256[] calldata amounts,
        BatchHint calldata hint
    ) external returns (bytes32 groupId);

    function executeRebalance(
        bytes32 rbId,
        Trade[] calldata sells,
        RebalanceBuy[] calldata buys,
        RebalanceTrigger trigger,
        BatchHint calldata hint
    ) external returns (bytes32 groupId);

}


// ===== FILE: src/interfaces/IFeeManager.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IFeeManager {
    function startAccrual() external;
    function accrueManagementFee(uint256 totalAssetsValue, uint256 effectiveSupply) external returns (uint256 shares);
    function claimManagementFee() external returns (uint256 shares);
    function claimPerformanceFee() external returns (uint256 shares);
    function claimAllFees() external returns (uint256 shares);
    function previewPerformanceFee(uint256 totalAssetsValue, uint256 effectiveSupply)
        external
        view
        returns (uint256 currentSharePrice, uint256 threshold, uint256 feeShares);
    function settlePerformanceFee(uint256 totalAssetsValue, uint256 effectiveSupply)
        external
        returns (uint256 shares, uint256 newHighWaterMark);
    function uncollectedFeeInShares() external view returns (uint256);
    function unclaimedManagementFeeShares() external view returns (uint256);
    function claimedManagementFeeShares() external view returns (uint256);
    function unclaimedPerformanceFeeShares() external view returns (uint256);
    function claimedPerformanceFeeShares() external view returns (uint256);
    function effectiveTotalSupply(uint256 totalSupply) external view returns (uint256);
}


// ===== FILE: src/interfaces/IERC7575.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC165} from "@openzeppelin/contracts/utils/introspection/IERC165.sol";
import { SignedPriceBundle } from "./Types.sol";


interface IERC7575 is IERC165 {
    /// @notice 返回该 Vault 对应的外部 ShareToken。
    function share() external view returns (address shareTokenAddress);
    function asset() external view returns (address assetTokenAddress);
    function totalAssets() external view returns (uint256 totalManagedAssets);
    function convertToShares(uint256 assets, SignedPriceBundle calldata priceBundle) external view returns (uint256 shares);
    function convertToAssets(uint256 shares, SignedPriceBundle calldata priceBundle) external view returns (uint256 assets);
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


// ===== FILE: src/interfaces/IRebalanceManager.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { RebalanceBuy, RebalanceTrigger, OndoTarget } from "./Types.sol";

interface IRebalanceManager {
    function setCoreVault(address coreVault) external;
    function setPassiveRebalanceInterval(uint256 interval) external;
    function prepareActiveRebalance(
        uint256 sellCount,
        uint256 buyCount,
        RebalanceBuy[] calldata buys,
        address[] calldata targetTokens,
        uint256[] calldata targetWeights
    ) external returns (bytes32 rbId);
    function preparePassiveRebalance(
        uint256 sellCount,
        uint256 buyCount,
        RebalanceTrigger trigger,
        uint8 riskLevel,
        bytes32 reasonHash,
        RebalanceBuy[] calldata buys,
        address[] calldata targetTokens,
        uint256[] calldata targetWeights
    ) external returns (bytes32 rbId);
    function validateRebalanceFilled(bytes32 rbId, address[] calldata tokens, uint256[] calldata amounts) external view;
    function settleRebalance(bytes32 rbId) external;
    function settleRebalanceFailure(bytes32 rbId) external;

    // Settlement entry points (called by AssetCustodian via CoreVault)
    function settleRebalanceComplete(
        bytes32 rbId,
        address[] calldata tokens,
        uint256[] calldata amounts
    ) external;

    function settleRebalanceFailed(bytes32 rbId) external;

    function isSettled(bytes32 rbId) external view returns (bool);

    // Pending Ondo Targets Management
    function savePendingOndoTargets(
        bytes32 rbId,
        address[] calldata targetTokens,
        uint256[] calldata targetWeights
    ) external;

    function getPendingOndoTargets(bytes32 rbId) external view returns (OndoTarget[] memory);

    function clearPendingOndoTargets(bytes32 rbId) external;

    function activeRebalanceId() external view returns (bytes32);
    function rebalanceTrigger(bytes32 rbId) external view returns (RebalanceTrigger);
    function rebalanceRiskLevel(bytes32 rbId) external view returns (uint8);
    function rebalanceReasonHash(bytes32 rbId) external view returns (bytes32);
    function minPassiveRebalanceInterval() external view returns (uint256);
    function lastPassiveRebalanceAt() external view returns (uint256);
}


// ===== FILE: src/interfaces/IRequestManager.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { RequestStatus } from "./Types.sol";

interface IRequestManager {
    struct RequestData {
        address owner;
        bool isDeposit;
        uint256 assets;
        uint256 shares;
        uint256 settledShares;
        uint256 refundAssets;
        uint256 navRound;
        uint256 totalAssetsSnapshot;
        uint256 supplySnapshot;
        RequestStatus status;
        bytes32 groupId;
    }

    struct DepositSettlement {
        address owner;
        uint256 spent;
        uint256 shares;
        uint256 refund;
    }

    struct RedeemSettlement {
        address owner;
        uint256 requestedShares;
        uint256 fulfilledShares;
        uint256 returnedShares;
        uint256 payout;
    }

    struct FailureSettlement {
        address owner;
        bool isDeposit;
        uint256 shares;
        uint256 refund;
    }

    function setCoreVault(address coreVault) external;
    function createDepositRequest(
        address owner,
        uint256 assets,
        uint256 shares,
        uint256 navRound,
        uint256 totalAssetsSnapshot,
        uint256 supplySnapshot
    ) external returns (uint256 reqId);
    function createRedeemRequest(
        address owner,
        uint256 shares,
        uint256 navRound,
        uint256 totalAssetsSnapshot,
        uint256 supplySnapshot,
        uint256 expectedAssets
    ) external returns (uint256 reqId);
    function setGroupId(uint256 reqId, bytes32 groupId) external;
    function validateDepositRequest(bytes32 groupId, uint256 reqId) external view;
    function settleDepositRequest(uint256 reqId, uint256 refundUsdc)
        external
        returns (DepositSettlement memory result);
    function validateRedeemRequest(bytes32 groupId, uint256 reqId) external view returns (uint256 totalShares);
    function settleRedeemRequest(
        uint256 reqId,
        uint256 usdcReceived
    ) external returns (RedeemSettlement memory result);
    function settleFailure(bytes32 groupId, uint256 reqId, uint256 returnedUsdc)
        external
        returns (FailureSettlement memory result);

    // Settlement entry points (called by AssetCustodian via CoreVault)
    function settleDeposit(
        bytes32 groupId,
        uint256 reqId,
        uint256 refundUsdc
    ) external returns (DepositSettlement memory result);

    function settleRedeem(
        bytes32 groupId,
        uint256 reqId,
        uint256 usdcReceived
    ) external returns (RedeemSettlement memory result);

    function settleRequestFailure(
        bytes32 groupId,
        uint256 reqId,
        uint256 returnedUsdc
    ) external returns (FailureSettlement memory result);

    function isSettled(bytes32 groupId) external view returns (bool);

    function getRequest(uint256 reqId) external view returns (RequestData memory);
    function userRequestCount(address owner) external view returns (uint256);
    function userRequestIdAt(address owner, uint256 index) external view returns (uint256);
    function userEscrowedShares(address owner) external view returns (uint256);
    function escrowedShares() external view returns (uint256);
}


// ===== FILE: src/interfaces/IShareToken.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC165} from "@openzeppelin/contracts/utils/introspection/IERC165.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";

interface IShareToken is IERC20, IERC165 {
    event VaultAdded(address indexed asset, address indexed vault);
    event VaultRemoved(address indexed asset, address indexed vault);

    function mint(address to, uint256 amount) external;
    function burn(address from, uint256 amount) external;
    function addVault(address asset, address vault_) external;
    function removeVault(address asset) external;
    function vault(address asset) external view returns (address);
    function asset(address vault_) external view returns (address);
    function isVault(address vault_) external view returns (bool);
}


// ===== FILE: src/interfaces/IWhitelist.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IWhitelist {
    function isWhitelisted(address account) external view returns (bool);
    function isAssetAllowed(address asset) external view returns (bool);
}


// ===== FILE: src/interfaces/IVaultSettlement.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IVaultSettlement {
    function settleDeposit(
        bytes32 groupId,
        uint256 reqId,
        address[] calldata etfTokens,
        uint256[] calldata amounts,
        uint256 refundUsdc,
        uint256 navRound
    ) external;

    function settleRedeem(bytes32 groupId, uint256 reqId, uint256 usdcReceived, uint256 navRound)
        external;
    function settleRebalance(bytes32 rbId, address[] calldata tokens, uint256[] calldata amounts)
        external;
    function settleRebalanceFailure(bytes32 rbId, bytes calldata reason) external;
    function settleFailure(bytes32 groupId, uint256 reqId, uint256 returnedUsdc, bytes calldata reason)
        external;
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


// ===== FILE: lib/openzeppelin-contracts/contracts/token/ERC20/ERC20.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "./IERC20.sol";
import {IERC20Metadata} from "./extensions/IERC20Metadata.sol";

contract ERC20 is IERC20, IERC20Metadata {
    string private _name;
    string private _symbol;
    uint256 private _totalSupply;
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    constructor(string memory name_, string memory symbol_) {
        _name = name_;
        _symbol = symbol_;
    }

    function name() public view virtual returns (string memory) {
        return _name;
    }

    function symbol() public view virtual returns (string memory) {
        return _symbol;
    }

    function decimals() public view virtual returns (uint8) {
        return 18;
    }

    function totalSupply() public view virtual returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view virtual returns (uint256) {
        return _balances[account];
    }

    function transfer(address to, uint256 value) public virtual returns (bool) {
        _transfer(msg.sender, to, value);
        return true;
    }

    function allowance(address owner, address spender) public view virtual returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 value) public virtual returns (bool) {
        _approve(msg.sender, spender, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) public virtual returns (bool) {
        uint256 currentAllowance = _allowances[from][msg.sender];
        require(currentAllowance >= value, "ERC20: insufficient allowance");
        unchecked {
            _approve(from, msg.sender, currentAllowance - value);
        }
        _transfer(from, to, value);
        return true;
    }

    function _transfer(address from, address to, uint256 value) internal virtual {
        require(from != address(0) && to != address(0), "ERC20: zero address");
        uint256 fromBalance = _balances[from];
        require(fromBalance >= value, "ERC20: insufficient balance");
        unchecked {
            _balances[from] = fromBalance - value;
            _balances[to] += value;
        }
        emit Transfer(from, to, value);
    }

    function _mint(address to, uint256 value) internal virtual {
        require(to != address(0), "ERC20: mint to zero");
        _totalSupply += value;
        unchecked {
            _balances[to] += value;
        }
        emit Transfer(address(0), to, value);
    }

    function _burn(address from, uint256 value) internal virtual {
        require(from != address(0), "ERC20: burn from zero");
        uint256 balance = _balances[from];
        require(balance >= value, "ERC20: burn exceeds balance");
        unchecked {
            _balances[from] = balance - value;
            _totalSupply -= value;
        }
        emit Transfer(from, address(0), value);
    }

    function _approve(address owner, address spender, uint256 value) internal virtual {
        require(owner != address(0) && spender != address(0), "ERC20: approve zero");
        _allowances[owner][spender] = value;
        emit Approval(owner, spender, value);
    }
}


// ===== FILE: lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/proxy/Proxy.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.5.0) (proxy/Proxy.sol)

pragma solidity ^0.8.20;

/**
 * @dev This abstract contract provides a fallback function that delegates all calls to another contract using the EVM
 * instruction `delegatecall`. We refer to the second contract as the _implementation_ behind the proxy, and it has to
 * be specified by overriding the virtual {_implementation} function.
 *
 * Additionally, delegation to the implementation can be triggered manually through the {_fallback} function, or to a
 * different contract through the {_delegate} function.
 *
 * The success and return data of the delegated call will be returned back to the caller of the proxy.
 */
abstract contract Proxy {
    /**
     * @dev Delegates the current call to `implementation`.
     *
     * This function does not return to its internal call site, it will return directly to the external caller.
     */
    function _delegate(address implementation) internal virtual {
        assembly {
            // Copy msg.data. We take full control of memory in this inline assembly
            // block because it will not return to Solidity code. We overwrite the
            // Solidity scratch pad at memory position 0.
            calldatacopy(0x00, 0x00, calldatasize())

            // Call the implementation.
            // out and outsize are 0 because we don't know the size yet.
            let result := delegatecall(gas(), implementation, 0x00, calldatasize(), 0x00, 0x00)

            // Copy the returned data.
            returndatacopy(0x00, 0x00, returndatasize())

            switch result
            // delegatecall returns 0 on error.
            case 0 {
                revert(0x00, returndatasize())
            }
            default {
                return(0x00, returndatasize())
            }
        }
    }

    /**
     * @dev This is a virtual function that should be overridden so it returns the address to which the fallback
     * function and {_fallback} should delegate.
     */
    function _implementation() internal view virtual returns (address);

    /**
     * @dev Delegates the current call to the address returned by `_implementation()`.
     *
     * This function does not return to its internal call site, it will return directly to the external caller.
     */
    function _fallback() internal virtual {
        _delegate(_implementation());
    }

    /**
     * @dev Fallback function that delegates calls to the address returned by `_implementation()`. Will run if no other
     * function in the contract matches the call data.
     */
    fallback() external payable virtual {
        _fallback();
    }
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


// ===== FILE: src/interfaces/IACCallback.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IACCallback {
    function onOrderSettled(bytes32 traceId, bool success, address token, uint256 amount, bytes calldata data) external;
}


// ===== FILE: src/interfaces/IOndoProxy.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IOndoProxy {
    struct DepositIntent {
        address receiver;
        address outputToken;
        uint256 destinationChainId;
        uint256 maxSlippageBps;
        uint256 maxLossPercentBps;
        bytes extraData;
    }

    function depositERC20(
        address token,
        uint256 amount,
        bytes32 traceId,
        DepositIntent calldata intent
    ) external;

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
