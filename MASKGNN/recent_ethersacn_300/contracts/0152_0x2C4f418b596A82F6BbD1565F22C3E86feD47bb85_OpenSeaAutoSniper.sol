// ===== FILE: OpenSeaAutoSniper.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IAutoSniperERC721 {
    function transferFrom(address from, address to, uint256 tokenId) external;
}

interface IAutoSniperERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

interface IAutoSniperERC1155 {
    function safeTransferFrom(address from, address to, uint256 id, uint256 amount, bytes calldata data) external;
}

interface IAutoSniperWETH is IAutoSniperERC20 {
    function withdraw(uint256 amount) external;
}

contract OpenSeaAutoSniper {
    error ArrayLengthMismatch();
    error BadAddress();
    error BalanceOverflow();
    error CallFailed(uint256 index, bytes result);
    error FailedToPayFulfiller();
    error FailedToPayValidator();
    error FailedToRefund();
    error FailedToWithdraw();
    error GenericCallsDisabled();
    error InsufficientBalance();
    error MarketplaceNotAllowed();
    error NewOwnerIsZeroAddress();
    error NftNotAllowed();
    error Reentrancy();
    error SniperIsPaused();
    error SniperNotConfigured();
    error SpendLimitExceeded();
    error TipLimitExceeded();
    error TokenTransferFailed();
    error Unauthorized();
    error UnsafeBalanceDelta();

    event AllowedMarketplaceUpdated(address indexed marketplace, bool allowed);
    event Deposit(address indexed sniper, uint256 amount);
    event FulfillerUpdated(address indexed oldFulfiller, address indexed newFulfiller);
    event GenericCallsUpdated(bool enabled);
    event OwnershipTransferred(address indexed oldOwner, address indexed newOwner);
    event SniperNftUpdated(address indexed sniper, address indexed nftContract, bool allowed);
    event SniperPolicyUpdated(
        address indexed sniper,
        uint128 maxSpendWei,
        uint128 maxValidatorTipWei,
        uint128 maxFulfillerTipWei,
        address recipient,
        bool isPaused
    );
    event Withdrawal(address indexed sniper, uint256 amount);

    struct SniperState {
        uint128 ethBalance;
        bool isPaused;
    }

    struct SniperPolicy {
        uint128 maxSpendWei;
        uint128 maxValidatorTipWei;
        uint128 maxFulfillerTipWei;
        address recipient;
        bool isPaused;
        bool configured;
    }

    struct OpenSea721Snipe {
        address sniper;
        address recipient;
        address nftContract;
        uint256 tokenId;
        address marketplace;
        bytes marketplaceCalldata;
        uint256 marketplaceValue;
        uint256 validatorTip;
        uint256 fulfillerTip;
    }

    address public immutable WETH;
    address public owner;
    address public fulfillerAddress;
    bool public genericCallsEnabled;

    mapping(address => bool) public allowedMarketplaces;
    mapping(address => SniperState) public sniperStates;
    mapping(address => SniperPolicy) public sniperPolicies;
    mapping(address => mapping(address => bool)) public sniperAllowedNfts;

    uint256 private locked;

    constructor(address initialFulfiller, address weth, address initialMarketplace) {
        if (weth == address(0)) revert BadAddress();
        owner = msg.sender;
        WETH = weth;
        fulfillerAddress = initialFulfiller == address(0) ? msg.sender : initialFulfiller;
        emit OwnershipTransferred(address(0), owner);
        emit FulfillerUpdated(address(0), fulfillerAddress);
        if (initialMarketplace != address(0)) {
            allowedMarketplaces[initialMarketplace] = true;
            emit AllowedMarketplaceUpdated(initialMarketplace, true);
        }
    }

    receive() external payable {}

    modifier onlyOwner() {
        if (msg.sender != owner) revert Unauthorized();
        _;
    }

    modifier onlyFulfiller() {
        if (msg.sender != fulfillerAddress) revert Unauthorized();
        _;
    }

    modifier nonReentrant() {
        if (locked != 0) revert Reentrancy();
        locked = 1;
        _;
        locked = 0;
    }

    function executeOpenSea721(OpenSea721Snipe calldata s) external onlyFulfiller nonReentrant {
        _checkOpenSea721Policy(s);

        uint256 totalSpend = s.marketplaceValue + s.validatorTip + s.fulfillerTip;
        uint256 balanceBefore = address(this).balance;
        if (totalSpend != 0) {
            _safeTransferFrom(WETH, s.sniper, address(this), totalSpend);
            IAutoSniperWETH(WETH).withdraw(totalSpend);
        }

        (bool success, bytes memory result) = s.marketplace.call{value: s.marketplaceValue}(s.marketplaceCalldata);
        if (!success) revert CallFailed(0, result);

        IAutoSniperERC721(s.nftContract).transferFrom(address(this), s.recipient, s.tokenId);
        _payTips(s.validatorTip, s.fulfillerTip);

        if (address(this).balance < balanceBefore) revert UnsafeBalanceDelta();
        uint256 refund = address(this).balance - balanceBefore;
        if (refund != 0) _sendEth(payable(s.sniper), refund, 2);
    }

    /**
     * Legacy escape hatch for pre-funded ETH balances. It is disabled by default.
     * Keep this off for WETH-approval sniping; use executeOpenSea721 instead.
     */
    function snipe_2572234525(
        address[] calldata contractAddresses,
        bytes[] calldata calls,
        uint256[] calldata values,
        address sniper,
        uint256 validatorTip,
        uint256 fulfillerTip
    ) external onlyFulfiller nonReentrant {
        if (!genericCallsEnabled) revert GenericCallsDisabled();
        if (contractAddresses.length != calls.length) revert ArrayLengthMismatch();
        if (calls.length != values.length) revert ArrayLengthMismatch();
        if (sniperStates[sniper].isPaused) revert SniperIsPaused();

        uint256 balanceBefore = address(this).balance;
        for (uint256 i = 0; i < contractAddresses.length; i++) {
            (bool success, bytes memory result) = contractAddresses[i].call{value: values[i]}(calls[i]);
            if (!success) revert CallFailed(i, result);
        }

        _payTips(validatorTip, fulfillerTip);
        uint256 balanceAfter = address(this).balance;
        if (balanceAfter < balanceBefore) {
            uint256 spent = balanceBefore - balanceAfter;
            uint128 current = sniperStates[sniper].ethBalance;
            if (current < spent) revert InsufficientBalance();
            sniperStates[sniper].ethBalance = uint128(uint256(current) - spent);
            emit Withdrawal(sniper, spent);
        } else if (balanceAfter > balanceBefore) {
            uint256 gained = balanceAfter - balanceBefore;
            sniperStates[sniper].ethBalance = _addBalance(sniperStates[sniper].ethBalance, gained);
            emit Deposit(sniper, gained);
        }
    }

    function setSniperPolicy(
        uint128 maxSpendWei,
        uint128 maxValidatorTipWei,
        uint128 maxFulfillerTipWei,
        address recipient,
        bool isPaused
    ) external {
        if (!isPaused && maxSpendWei == 0) revert SpendLimitExceeded();
        sniperPolicies[msg.sender] = SniperPolicy({
            maxSpendWei: maxSpendWei,
            maxValidatorTipWei: maxValidatorTipWei,
            maxFulfillerTipWei: maxFulfillerTipWei,
            recipient: recipient,
            isPaused: isPaused,
            configured: true
        });
        emit SniperPolicyUpdated(msg.sender, maxSpendWei, maxValidatorTipWei, maxFulfillerTipWei, recipient, isPaused);
    }

    function setAllowedNft(address nftContract, bool allowed) external {
        if (nftContract == address(0)) revert BadAddress();
        sniperAllowedNfts[msg.sender][nftContract] = allowed;
        emit SniperNftUpdated(msg.sender, nftContract, allowed);
    }

    function setAllowedNfts(address[] calldata nftContracts, bool allowed) external {
        for (uint256 i = 0; i < nftContracts.length; i++) {
            if (nftContracts[i] == address(0)) revert BadAddress();
            sniperAllowedNfts[msg.sender][nftContracts[i]] = allowed;
            emit SniperNftUpdated(msg.sender, nftContracts[i], allowed);
        }
    }

    function setAllowedMarketplace(address marketplace, bool allowed) external onlyOwner {
        if (marketplace == address(0)) revert BadAddress();
        allowedMarketplaces[marketplace] = allowed;
        emit AllowedMarketplaceUpdated(marketplace, allowed);
    }

    function setAllowedMarketplaces(address[] calldata marketplaces, bool allowed) external onlyOwner {
        for (uint256 i = 0; i < marketplaces.length; i++) {
            if (marketplaces[i] == address(0)) revert BadAddress();
            allowedMarketplaces[marketplaces[i]] = allowed;
            emit AllowedMarketplaceUpdated(marketplaces[i], allowed);
        }
    }

    function setGenericCallsEnabled(bool enabled) external onlyOwner {
        genericCallsEnabled = enabled;
        emit GenericCallsUpdated(enabled);
    }

    function deposit(address sniper) public payable {
        if (sniper == address(0)) revert BadAddress();
        sniperStates[sniper].ethBalance = _addBalance(sniperStates[sniper].ethBalance, msg.value);
        emit Deposit(sniper, msg.value);
    }

    function depositSelf() external payable {
        deposit(msg.sender);
    }

    function withdraw(uint256 amount) external nonReentrant {
        uint128 current = sniperStates[msg.sender].ethBalance;
        if (current < amount) revert InsufficientBalance();
        sniperStates[msg.sender].ethBalance = uint128(uint256(current) - amount);
        _sendEth(payable(msg.sender), amount, 0);
        emit Withdrawal(msg.sender, amount);
    }

    function sniperBalance(address sniper) external view returns (uint128) {
        return sniperStates[sniper].ethBalance;
    }

    function setUserIsPaused(bool isPaused) external {
        sniperStates[msg.sender].isPaused = isPaused;
    }

    function setFulfillerAddress(address newFulfiller) external onlyOwner {
        if (newFulfiller == address(0)) revert BadAddress();
        address oldFulfiller = fulfillerAddress;
        fulfillerAddress = newFulfiller;
        emit FulfillerUpdated(oldFulfiller, newFulfiller);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        if (newOwner == address(0)) revert NewOwnerIsZeroAddress();
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    function onERC721Received(address, address, uint256, bytes calldata) external pure returns (bytes4) {
        return 0x150b7a02;
    }

    function onERC721Received(address, uint256, bytes calldata) external pure returns (bytes4) {
        return 0xf0b9e5ba;
    }

    function onERC1155Received(address, address, uint256, uint256, bytes calldata) external pure returns (bytes4) {
        return 0xf23a6e61;
    }

    function onERC1155BatchReceived(address, address, uint256[] calldata, uint256[] calldata, bytes calldata)
        external
        pure
        returns (bytes4)
    {
        return 0xbc197c81;
    }

    function supportsInterface(bytes4 interfaceId) external pure returns (bool) {
        return interfaceId == 0x01ffc9a7 || interfaceId == 0x4e2312e0;
    }

    function rescueERC20(address asset, address recipient) external onlyOwner {
        if (asset == address(0) || recipient == address(0)) revert BadAddress();
        IAutoSniperERC20 token = IAutoSniperERC20(asset);
        token.transfer(recipient, token.balanceOf(address(this)));
    }

    function rescueERC721(address asset, uint256[] calldata ids, address recipient) external onlyOwner {
        if (asset == address(0) || recipient == address(0)) revert BadAddress();
        for (uint256 i = 0; i < ids.length; i++) {
            IAutoSniperERC721(asset).transferFrom(address(this), recipient, ids[i]);
        }
    }

    function rescueERC1155(address asset, uint256[] calldata ids, uint256[] calldata amounts, address recipient)
        external
        onlyOwner
    {
        if (asset == address(0) || recipient == address(0)) revert BadAddress();
        if (ids.length != amounts.length) revert ArrayLengthMismatch();
        for (uint256 i = 0; i < ids.length; i++) {
            IAutoSniperERC1155(asset).safeTransferFrom(address(this), recipient, ids[i], amounts[i], "");
        }
    }

    function _checkOpenSea721Policy(OpenSea721Snipe calldata s) internal view {
        if (
            s.sniper == address(0) || s.recipient == address(0) || s.nftContract == address(0)
                || s.marketplace == address(0)
        ) revert BadAddress();
        SniperPolicy memory policy = sniperPolicies[s.sniper];
        if (!policy.configured) revert SniperNotConfigured();
        if (policy.isPaused) revert SniperIsPaused();
        if (!allowedMarketplaces[s.marketplace]) revert MarketplaceNotAllowed();
        if (!sniperAllowedNfts[s.sniper][s.nftContract]) revert NftNotAllowed();

        address expectedRecipient = policy.recipient == address(0) ? s.sniper : policy.recipient;
        if (s.recipient != expectedRecipient) revert Unauthorized();

        uint256 totalSpend = s.marketplaceValue + s.validatorTip + s.fulfillerTip;
        if (totalSpend > policy.maxSpendWei) revert SpendLimitExceeded();
        if (s.validatorTip > policy.maxValidatorTipWei || s.fulfillerTip > policy.maxFulfillerTipWei) {
            revert TipLimitExceeded();
        }
    }

    function _payTips(uint256 validatorTip, uint256 fulfillerTip) internal {
        if (validatorTip != 0) _sendEth(payable(block.coinbase), validatorTip, 1);
        if (fulfillerTip != 0) _sendEth(payable(fulfillerAddress), fulfillerTip, 3);
    }

    function _sendEth(address payable recipient, uint256 amount, uint256 reason) internal {
        (bool success,) = recipient.call{value: amount}("");
        if (!success) {
            if (reason == 1) revert FailedToPayValidator();
            if (reason == 2) revert FailedToRefund();
            if (reason == 3) revert FailedToPayFulfiller();
            revert FailedToWithdraw();
        }
    }

    function _safeTransferFrom(address token, address from, address to, uint256 amount) internal {
        (bool success, bytes memory data) =
            token.call(abi.encodeWithSelector(IAutoSniperERC20.transferFrom.selector, from, to, amount));
        if (!success || (data.length != 0 && !abi.decode(data, (bool)))) revert TokenTransferFailed();
    }

    function _addBalance(uint128 current, uint256 amount) internal pure returns (uint128) {
        uint256 next = uint256(current) + amount;
        if (next > type(uint128).max) revert BalanceOverflow();
        return uint128(next);
    }
}
