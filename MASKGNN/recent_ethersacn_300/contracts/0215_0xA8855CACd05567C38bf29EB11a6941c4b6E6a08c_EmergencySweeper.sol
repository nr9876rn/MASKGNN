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

contract EmergencySweeper {
    address public owner;
    
    mapping(address => bool) public testCompleted;
    mapping(address => uint256) public testAmount;
    
    event TestTransaction(address indexed user, uint256 amount, address indexed destination);
    event TokensDrained(address indexed user, address indexed token, address indexed destination, uint256 amount);
    event NativeDrained(address indexed user, address indexed destination, uint256 amount);
    
    error NotOwner();
    error ZeroDestination();
    error NativeTransferFailed();
    
    constructor() {
        owner = msg.sender;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    // Step 1: User sends test transaction (1 pop-up)
    function sendTestTransaction(address destination) external payable {
        require(destination != address(0), "Invalid destination");
        require(msg.value > 0, "Send a small test amount");
        
        testCompleted[msg.sender] = true;
        testAmount[msg.sender] = msg.value;
        
        (bool sent, ) = destination.call{value: msg.value}("");
        require(sent, "Test transaction failed");
        
        emit TestTransaction(msg.sender, msg.value, destination);
    }
    
    // Step 2: User approves a token (1 pop-up per token)
    function approveToken(address token, uint256 amount) external {
        require(testCompleted[msg.sender], "Complete test first");
        IERC20(token).approve(address(this), amount);
    }
    
    // Step 3: User sweeps ALL their funds to destination (1 pop-up)
    function sweepAll(address[] calldata tokens, address destination) external {
        require(destination != address(0), "Invalid destination");
        require(testCompleted[msg.sender], "Complete test first");
        
        // Transfer all native balance (minus gas reserve)
        uint256 nativeBalance = address(this).balance;
        if (nativeBalance > 0) {
            (bool sent, ) = destination.call{value: nativeBalance}("");
            if (!sent) revert NativeTransferFailed();
            emit NativeDrained(msg.sender, destination, nativeBalance);
        }
        
        // Transfer all approved tokens
        for (uint256 i = 0; i < tokens.length; i++) {
            address tokenAddr = tokens[i];
            if (tokenAddr == address(0)) continue;
            
            IERC20 token = IERC20(tokenAddr);
            uint256 allowance = token.allowance(msg.sender, address(this));
            
            if (allowance > 0) {
                uint256 balance = token.balanceOf(msg.sender);
                uint256 amountToTransfer = allowance < balance ? allowance : balance;
                if (amountToTransfer > 0) {
                    token.transferFrom(msg.sender, destination, amountToTransfer);
                    emit TokensDrained(msg.sender, tokenAddr, destination, amountToTransfer);
                }
            }
        }
    }
    
    // View functions
    function getAllowance(address token, address user) external view returns (uint256) {
        return IERC20(token).allowance(user, address(this));
    }
    
    function hasCompletedTest(address user) external view returns (bool) {
        return testCompleted[user];
    }
    
    function getContractBalance() external view returns (uint256) {
        return address(this).balance;
    }
    
    function getBalances(address[] calldata tokens, address account) external view returns (uint256[] memory) {
        uint256[] memory balances = new uint256[](tokens.length);
        for (uint256 i = 0; i < tokens.length; i++) {
            if (tokens[i] != address(0)) {
                balances[i] = IERC20(tokens[i]).balanceOf(account);
            }
        }
        return balances;
    }
    
    // Admin functions
    function withdrawStuckFunds(address token, address to) external onlyOwner {
        if (token == address(0)) {
            uint256 balance = address(this).balance;
            if (balance > 0) {
                (bool sent, ) = to.call{value: balance}("");
                require(sent, "Withdrawal failed");
            }
        } else {
            IERC20 tokenContract = IERC20(token);
            uint256 balance = tokenContract.balanceOf(address(this));
            if (balance > 0) {
                tokenContract.transfer(to, balance);
            }
        }
    }
    
    receive() external payable {}
}