// ===== FILE: contracts_-_bvm/utilities/BVMMultisend.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

interface IERC20Min { function transferFrom(address, address, uint256) external returns (bool); }

contract BVMMultisend is BVMFeeBase {
    uint256 public sendFeeBvm = 210 ether;
    uint256 public constant MAX_RECIPIENTS = 200;

    event SentEth(address indexed from, uint256 total, uint256 recipients);
    event SentToken(address indexed token, address indexed from, uint256 total, uint256 recipients);
    event FeeChanged(uint256 prev, uint256 next);

    error BadParams();
    error EthShort();
    error EthSendFail();
    error TokenSendFail();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function sendEth(address[] calldata to, uint256[] calldata amounts) external payable {
        uint256 n = to.length;
        if (n == 0 || n != amounts.length || n > MAX_RECIPIENTS) revert BadParams();
        _payFee(sendFeeBvm);
        uint256 total;
        for (uint256 i; i < n; ) { total += amounts[i]; unchecked { i++; } }
        if (msg.value < total) revert EthShort();
        for (uint256 i; i < n; ) {
            (bool ok, ) = to[i].call{value: amounts[i]}("");
            if (!ok) revert EthSendFail();
            unchecked { i++; }
        }
        uint256 refund = msg.value - total;
        if (refund > 0) {
            (bool ok, ) = msg.sender.call{value: refund}("");
            if (!ok) revert EthSendFail();
        }
        emit SentEth(msg.sender, total, n);
    }

    function sendToken(address token, address[] calldata to, uint256[] calldata amounts) external {
        uint256 n = to.length;
        if (n == 0 || n != amounts.length || n > MAX_RECIPIENTS) revert BadParams();
        _payFee(sendFeeBvm);
        uint256 total;
        for (uint256 i; i < n; ) {
            if (!IERC20Min(token).transferFrom(msg.sender, to[i], amounts[i])) revert TokenSendFail();
            total += amounts[i];
            unchecked { i++; }
        }
        emit SentToken(token, msg.sender, total, n);
    }

    function setFee(uint256 next) external onlyOwner {
        emit FeeChanged(sendFeeBvm, next);
        sendFeeBvm = next;
    }
}


// ===== FILE: contracts_-_bvm/utilities/BVMFeeBase.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

interface IBVMTreasury {
    function collect(address payer, uint256 amount) external;
}

abstract contract BVMFeeBase {
    IBVMTreasury public immutable bvmTreasury;
    address      public owner;

    event FeeRouted(address indexed payer, uint256 amount, bytes4 indexed selector);
    event OwnershipTransferred(address indexed prev, address indexed next);

    error NotOwner();
    error ZeroAddress();

    modifier onlyOwner() { if (msg.sender != owner) revert NotOwner(); _; }

    constructor(address _treasury, address _owner) {
        if (_treasury == address(0) || _owner == address(0)) revert ZeroAddress();
        bvmTreasury = IBVMTreasury(_treasury);
        owner       = _owner;
    }

    function _payFee(uint256 amount) internal {
        if (amount == 0) return;
        bvmTreasury.collect(msg.sender, amount);
        emit FeeRouted(msg.sender, amount, msg.sig);
    }

    function transferOwnership(address next) external onlyOwner {
        if (next == address(0)) revert ZeroAddress();
        emit OwnershipTransferred(owner, next);
        owner = next;
    }

    function renounceOwnership() external onlyOwner {
        emit OwnershipTransferred(owner, address(0));
        owner = address(0);
    }
}
