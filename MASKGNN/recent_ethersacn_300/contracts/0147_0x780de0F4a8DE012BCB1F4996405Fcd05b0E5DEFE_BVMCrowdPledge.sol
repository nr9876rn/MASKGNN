// ===== FILE: contracts_-_bvm/utilities/BVMCrowdPledge.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

contract BVMCrowdPledge is BVMFeeBase {
    uint256 public launchFeeBvm = 500 ether;

    struct Campaign {
        address creator;
        uint128 goalWei;
        uint128 raisedWei;
        uint64  endsAt;
        bool    finalized;
        bool    succeeded;
        bytes32 brief;
        string  detailsCid;
    }
    Campaign[] public campaigns;
    mapping(uint256 => mapping(address => uint128)) public pledgedBy;

    event Launched(uint256 indexed id, address indexed creator, uint128 goal, uint64 endsAt, bytes32 brief, string detailsCid);
    event Pledged(uint256 indexed id, address indexed pledger, uint128 amount, uint128 newTotal);
    event Finalized(uint256 indexed id, bool succeeded, uint128 raised);
    event Refunded(uint256 indexed id, address indexed pledger, uint128 amount);
    event FeeChanged(uint256 prev, uint256 next);

    error BadParams();
    error CampaignClosed();
    error NotEnded();
    error AlreadyFinal();
    error NoFunds();
    error EthSendFail();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function launch(uint128 goalWei, uint64 endsAt, bytes32 brief, string calldata detailsCid) external returns (uint256 id) {
        if (goalWei == 0 || endsAt <= block.timestamp) revert BadParams();
        _payFee(launchFeeBvm);
        id = campaigns.length;
        campaigns.push();
        Campaign storage c = campaigns[id];
        c.creator = msg.sender; c.goalWei = goalWei; c.endsAt = endsAt;
        c.brief = brief; c.detailsCid = detailsCid;
        emit Launched(id, msg.sender, goalWei, endsAt, brief, detailsCid);
    }

    function pledge(uint256 id) external payable {
        Campaign storage c = campaigns[id];
        if (block.timestamp > c.endsAt || c.finalized) revert CampaignClosed();
        if (msg.value == 0) revert BadParams();
        pledgedBy[id][msg.sender] += uint128(msg.value);
        c.raisedWei += uint128(msg.value);
        emit Pledged(id, msg.sender, uint128(msg.value), c.raisedWei);
    }

    function finalize(uint256 id) external {
        Campaign storage c = campaigns[id];
        if (c.finalized) revert AlreadyFinal();
        if (block.timestamp <= c.endsAt) revert NotEnded();
        c.finalized = true;
        c.succeeded = c.raisedWei >= c.goalWei;
        if (c.succeeded) {
            (bool ok, ) = c.creator.call{value: c.raisedWei}("");
            if (!ok) revert EthSendFail();
        }
        emit Finalized(id, c.succeeded, c.raisedWei);
    }

    function refund(uint256 id) external {
        Campaign storage c = campaigns[id];
        if (!c.finalized || c.succeeded) revert CampaignClosed();
        uint128 amt = pledgedBy[id][msg.sender];
        if (amt == 0) revert NoFunds();
        pledgedBy[id][msg.sender] = 0;
        (bool ok, ) = msg.sender.call{value: amt}("");
        if (!ok) revert EthSendFail();
        emit Refunded(id, msg.sender, amt);
    }

    function totalCampaigns() external view returns (uint256) { return campaigns.length; }

    function setFee(uint256 next) external onlyOwner {
        emit FeeChanged(launchFeeBvm, next);
        launchFeeBvm = next;
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
