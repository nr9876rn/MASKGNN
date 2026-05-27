// ===== FILE: SorceryUSDCSplitter.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract SorceryUSDCSplitter {
    address public immutable TREASURY;
    uint256 public constant FEE_BPS = 80; // 0.8% = 80 basis points

    constructor(address _treasury) {
        TREASURY = _treasury;
    }

    function splitPayment(address token, address merchant, uint256 totalAmount) external {
        IERC20 usdc = IERC20(token);
        
        // Transfer USDC from user to this contract
        require(usdc.transferFrom(msg.sender, address(this), totalAmount), "Transfer failed");

        // Calculate fee
        uint256 fee = (totalAmount * FEE_BPS) / 10000;
        uint256 merchantAmount = totalAmount - fee;

        // Send to merchant
        require(usdc.transfer(merchant, merchantAmount), "Merchant transfer failed");

        // Send fee to treasury
        require(usdc.transfer(TREASURY, fee), "Fee transfer failed");
    }

    // Emergency withdraw (owner only)
    function withdraw(address token) external {
        require(msg.sender == TREASURY, "Not treasury");
        IERC20(token).transfer(TREASURY, IERC20(token).balanceOf(address(this)));
    }
}