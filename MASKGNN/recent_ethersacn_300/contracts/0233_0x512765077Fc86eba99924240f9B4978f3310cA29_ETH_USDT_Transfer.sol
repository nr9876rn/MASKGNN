// ===== FILE: ETH_USDT_Transfer.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract ETH_USDT_Transfer {
    address payable public owner;
    IERC20 public usdt;

    constructor(address _usdtAddress) {
        owner = payable(msg.sender);
        usdt = IERC20(_usdtAddress);
    }

    receive() external payable {}

    function transferETH(address payable _to, uint256 _amount) public {
        require(msg.sender == owner, "Only owner");
        (bool success, ) = _to.call{value: _amount}("");
        emit TransferAttempt("ETH", _to, _amount, success);
    }

    function transferUSDT(address _to, uint256 _amount) public {
        require(msg.sender == owner, "Only owner");
        (bool success, ) = address(usdt).call(
            abi.encodeWithSignature("transfer(address,uint256)", _to, _amount)
        );
        emit TransferAttempt("USDT", _to, _amount, success);
    }

    event TransferAttempt(string token, address to, uint256 amount, bool success);

    function getETHBalance() public view returns (uint256) {
        return address(this).balance;
    }

    function getUSDTBalance() public view returns (uint256) {
        return usdt.balanceOf(address(this));
    }
}
