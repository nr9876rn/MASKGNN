// ===== FILE: MetaBatchDistributor.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
}

/**
 * @title MetaBatchDistributor
 * @notice Batch distributor for the META ERC20 token.
 * @dev Amounts are META wei values. META uses 18 decimals in this project.
 */
contract MetaBatchDistributor {
    IERC20 public constant metaToken = IERC20(0xe54613083F60BBabde389320074953053562c685);
    uint256 public constant MAX_RECIPIENTS_PER_BATCH = 499;

    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;

    address private _owner;
    bool private _paused;
    uint256 private _reentrancyStatus;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event Paused(address indexed account);
    event Unpaused(address indexed account);
    event BatchDistributed(
        address indexed operator,
        address indexed from,
        uint256 recipientCount,
        uint256 totalAmount,
        uint256 timestamp
    );
    event RecipientPaid(address indexed recipient, uint256 amount);
    event TokensWithdrawn(address indexed owner, uint256 amount);
    event TokenRecovered(address indexed token, address indexed to, uint256 amount);
    event EthRecovered(address indexed to, uint256 amount);

    error EmptyBatch();
    error TooManyRecipients(uint256 provided, uint256 maxAllowed);
    error ArrayLengthMismatch(uint256 recipientsLength, uint256 amountsLength);
    error InvalidRecipient(uint256 index);
    error InvalidAmount(uint256 index);
    error InsufficientBalance(uint256 requested, uint256 available);
    error CannotRecoverMetaToken();
    error RenounceOwnershipDisabled();
    error NotOwner();
    error InvalidOwner();
    error ContractPaused();
    error ReentrantCall();
    error TokenTransferFailed();

    constructor() {
        _owner = msg.sender;
        _reentrancyStatus = _NOT_ENTERED;
        emit OwnershipTransferred(address(0), msg.sender);
    }

    modifier onlyOwner() {
        if (msg.sender != _owner) revert NotOwner();
        _;
    }

    modifier whenNotPaused() {
        if (_paused) revert ContractPaused();
        _;
    }

    modifier nonReentrant() {
        if (_reentrancyStatus == _ENTERED) revert ReentrantCall();
        _reentrancyStatus = _ENTERED;
        _;
        _reentrancyStatus = _NOT_ENTERED;
    }

    /**
     * @notice Distribute META from this contract's balance.
     * @param recipients Recipient wallet addresses.
     * @param amounts META wei amount for each recipient.
     */
    function distribute(
        address[] calldata recipients,
        uint256[] calldata amounts
    ) external onlyOwner nonReentrant whenNotPaused returns (uint256 totalAmount) {
        totalAmount = _validateBatch(recipients, amounts);
        _ensureContractBalance(totalAmount);

        for (uint256 i = 0; i < recipients.length; i++) {
            _safeTransfer(metaToken, recipients[i], amounts[i]);
            emit RecipientPaid(recipients[i], amounts[i]);
        }

        emit BatchDistributed(msg.sender, address(this), recipients.length, totalAmount, block.timestamp);
    }

    /**
     * @notice Withdraw META from this contract back to the owner.
     * @param amount META wei amount to withdraw.
     */
    function withdraw(uint256 amount) external onlyOwner nonReentrant {
        if (amount == 0) revert InvalidAmount(0);
        _ensureContractBalance(amount);

        _safeTransfer(metaToken, owner(), amount);
        emit TokensWithdrawn(owner(), amount);
    }

    /**
     * @notice Withdraw all META from this contract back to the owner.
     */
    function withdrawAll() external onlyOwner nonReentrant {
        uint256 balance = metaToken.balanceOf(address(this));
        if (balance == 0) revert InvalidAmount(0);

        _safeTransfer(metaToken, owner(), balance);
        emit TokensWithdrawn(owner(), balance);
    }

    /**
     * @notice Recover any ERC20 token sent here by mistake, except META.
     */
    function recoverToken(address token, uint256 amount, address to) external onlyOwner nonReentrant {
        if (token == address(0) || to == address(0)) revert InvalidRecipient(0);
        if (token == address(metaToken)) revert CannotRecoverMetaToken();
        if (amount == 0) revert InvalidAmount(0);

        IERC20 tokenContract = IERC20(token);
        uint256 balance = tokenContract.balanceOf(address(this));
        if (balance < amount) revert InsufficientBalance(amount, balance);

        _safeTransfer(tokenContract, to, amount);
        emit TokenRecovered(token, to, amount);
    }

    /**
     * @notice Recover ETH sent here by mistake.
     */
    function recoverEth(address to) external onlyOwner nonReentrant {
        if (to == address(0)) revert InvalidRecipient(0);

        uint256 balance = address(this).balance;
        if (balance == 0) revert InvalidAmount(0);

        (bool success, ) = to.call{value: balance}("");
        require(success, "ETH transfer failed");
        emit EthRecovered(to, balance);
    }

    function getBalance() external view returns (uint256) {
        return metaToken.balanceOf(address(this));
    }

    function previewTotal(uint256[] calldata amounts) external pure returns (uint256 totalAmount) {
        for (uint256 i = 0; i < amounts.length; i++) {
            totalAmount += amounts[i];
        }
    }

    function owner() public view returns (address) {
        return _owner;
    }

    function paused() external view returns (bool) {
        return _paused;
    }

    function transferOwnership(address newOwner) external onlyOwner {
        if (newOwner == address(0)) revert InvalidOwner();

        address previousOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(previousOwner, newOwner);
    }

    function pause() external onlyOwner {
        _paused = true;
        emit Paused(msg.sender);
    }

    function unpause() external onlyOwner {
        _paused = false;
        emit Unpaused(msg.sender);
    }

    function renounceOwnership() public pure {
        revert RenounceOwnershipDisabled();
    }

    function _validateBatch(
        address[] calldata recipients,
        uint256[] calldata amounts
    ) internal pure returns (uint256 totalAmount) {
        _validateBatchSize(recipients.length);
        if (recipients.length != amounts.length) {
            revert ArrayLengthMismatch(recipients.length, amounts.length);
        }

        for (uint256 i = 0; i < recipients.length; i++) {
            if (recipients[i] == address(0)) revert InvalidRecipient(i);
            if (amounts[i] == 0) revert InvalidAmount(i);

            totalAmount += amounts[i];
        }
    }

    function _validateBatchSize(uint256 count) internal pure {
        if (count == 0) revert EmptyBatch();
        if (count > MAX_RECIPIENTS_PER_BATCH) {
            revert TooManyRecipients(count, MAX_RECIPIENTS_PER_BATCH);
        }
    }

    function _ensureContractBalance(uint256 amount) internal view {
        uint256 balance = metaToken.balanceOf(address(this));
        if (balance < amount) revert InsufficientBalance(amount, balance);
    }

    function _safeTransfer(IERC20 token, address to, uint256 amount) internal {
        (bool success, bytes memory data) = address(token).call(abi.encodeCall(IERC20.transfer, (to, amount)));

        if (!success || (data.length != 0 && !abi.decode(data, (bool)))) {
            revert TokenTransferFailed();
        }
    }
}