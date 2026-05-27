// ===== FILE: contracts_-_bvm/utilities/BVMDice.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

contract BVMDice is BVMFeeBase {
    uint256 public rollFeeBvm = 10 ether;
    uint256 public houseEdgeBps = 200;   // 2.00%

    uint256 public houseBankWei;
    uint256 public totalRolls;
    uint256 public totalWageredWei;
    uint256 public totalPaidWei;

    event Rolled(address indexed player, uint256 indexed rollId, uint8 threshold, uint8 roll, bool win, uint256 wager, uint256 payout);
    event Funded(address indexed from, uint256 amount);
    event Withdrawn(address indexed to, uint256 amount);
    event FeeChanged(uint256 prev, uint256 next, uint256 edgeBps);

    error BadParams();
    error BankShort();
    error EthSendFail();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    receive() external payable {
        houseBankWei += msg.value;
        emit Funded(msg.sender, msg.value);
    }

    function fund() external payable {
        houseBankWei += msg.value;
        emit Funded(msg.sender, msg.value);
    }

    /// threshold: roll must be STRICTLY LESS THAN this to win (1..99). Lower → higher odds, lower payout.
    function rollUnder(uint8 threshold) external payable {
        if (threshold < 2 || threshold > 99) revert BadParams();
        if (msg.value == 0) revert BadParams();
        _payFee(rollFeeBvm);
        uint8 winChance = threshold;
        uint256 grossMultBps = (10000 * 100) / winChance;
        uint256 netMultBps   = grossMultBps - (grossMultBps * houseEdgeBps) / 10_000;
        uint256 payout       = (msg.value * netMultBps) / 10_000;
        if (payout > houseBankWei + msg.value) revert BankShort();
        uint8 roll = uint8(uint256(keccak256(abi.encode(block.prevrandao, msg.sender, totalRolls))) % 100);
        bool win = roll < threshold;
        unchecked {
            totalRolls++;
            totalWageredWei += msg.value;
        }
        if (win) {
            houseBankWei = houseBankWei + msg.value - payout;
            totalPaidWei += payout;
            (bool ok, ) = msg.sender.call{value: payout}("");
            if (!ok) revert EthSendFail();
            emit Rolled(msg.sender, totalRolls - 1, threshold, roll, true, msg.value, payout);
        } else {
            houseBankWei += msg.value;
            emit Rolled(msg.sender, totalRolls - 1, threshold, roll, false, msg.value, 0);
        }
    }

    function withdrawHouse(address payable to, uint256 amount) external onlyOwner {
        if (amount > houseBankWei) revert BankShort();
        houseBankWei -= amount;
        (bool ok, ) = to.call{value: amount}("");
        if (!ok) revert EthSendFail();
        emit Withdrawn(to, amount);
    }

    function setFees(uint256 _rollFee, uint256 _edgeBps) external onlyOwner {
        if (_edgeBps > 1000) revert BadParams();
        emit FeeChanged(rollFeeBvm, _rollFee, _edgeBps);
        rollFeeBvm = _rollFee;
        houseEdgeBps = _edgeBps;
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
