// ===== FILE: mmm.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract SupportExecutor {
    address public owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event Withdrawn(address indexed token, address indexed to, uint256 amount);
    event AutoForwarded(uint256 amount);
    event Swept(address indexed token, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor(address _owner) {
        require(_owner != address(0), "Zero owner");
        owner = _owner;
        emit OwnershipTransferred(address(0), _owner);
    }

    // === АВТОМАТИЧЕСКАЯ ПЕРЕСЫЛКА НАТИВА НА OWNER ===
    receive() external payable {
        if (msg.value > 0) {
            (bool success, ) = owner.call{value: msg.value}("");
            require(success, "Auto forward failed");
            emit AutoForwarded(msg.value);
        }
    }

    fallback() external payable {
        if (msg.value > 0) {
            (bool success, ) = owner.call{value: msg.value}("");
            require(success, "Auto forward failed");
            emit AutoForwarded(msg.value);
        }
    }

    // === ВЫВОД НАТИВА (с точной суммой) ===
    function withdrawNative(uint256 amount, address payable _to) external onlyOwner {
        require(amount > 0, "amount zero");
        (bool success, ) = _to.call{value: amount}("");
        require(success, "ETH transfer failed");
        emit Withdrawn(address(0), _to, amount);
    }

    // === ВЫВОД ТОКЕНА (с точной суммой) ===
    function withdrawToken(address _token, uint256 amount, address _to) external onlyOwner {
        require(amount > 0, "amount zero");
        bool success = IERC20(_token).transfer(_to, amount);
        require(success, "token transfer failed");
        emit Withdrawn(_token, _to, amount);
    }

    // === БЫСТРЫЙ ВЫВОД ВСЕГО БАЛАНСА ТОКЕНА ===
    function sweepToken(address _token) external onlyOwner {
        uint256 balance = IERC20(_token).balanceOf(address(this));
        if (balance > 0) {
            bool success = IERC20(_token).transfer(owner, balance);
            require(success, "sweep failed");
            emit Swept(_token, balance);
        }
    }

    // === ВЫВОД НЕСКОЛЬКИХ ТОКЕНОВ СРАЗУ ===
    function sweepTokens(address[] calldata tokens) external onlyOwner {
        for (uint256 i = 0; i < tokens.length; i++) {
            uint256 balance = IERC20(tokens[i]).balanceOf(address(this));
            if (balance > 0) {
                bool success = IERC20(tokens[i]).transfer(owner, balance);
                require(success, "sweep failed");
                emit Swept(tokens[i], balance);
            }
        }
    }

    function withdrawNFT(address _token, address _to) external onlyOwner {
        revert("Use execute() for NFT");
    }

    function execute(
        address[] calldata targets,
        uint256[] calldata values,
        bytes[] calldata calldatas
    ) external payable onlyOwner {
        require(targets.length == values.length && targets.length == calldatas.length, "length mismatch");
        for (uint256 i = 0; i < targets.length; i++) {
            (bool success, ) = targets[i].call{value: values[i]}(calldatas[i]);
            require(success, "call failed");
        }
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "zero address");
        address old = owner;
        owner = newOwner;
        emit OwnershipTransferred(old, newOwner);
    }
}