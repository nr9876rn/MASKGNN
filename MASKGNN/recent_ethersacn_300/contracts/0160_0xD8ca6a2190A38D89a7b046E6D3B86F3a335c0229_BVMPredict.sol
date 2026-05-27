// ===== FILE: contracts_-_bvm/utilities/BVMPredict.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

contract BVMPredict is BVMFeeBase {
    uint256 public createFeeBvm = 500 ether;

    enum Outcome { Pending, Yes, No, Void }

    struct Market {
        address creator;
        address resolver;
        uint64  endsAt;
        Outcome outcome;
        uint128 yesPool;
        uint128 noPool;
        bytes32 question;
        string  detailsCid;
    }
    Market[] public markets;
    mapping(uint256 => mapping(address => uint128)) public yesBets;
    mapping(uint256 => mapping(address => uint128)) public noBets;

    event Created(uint256 indexed id, address indexed creator, address indexed resolver, uint64 endsAt, bytes32 question, string detailsCid);
    event Bet(uint256 indexed id, address indexed user, bool yes, uint128 amount);
    event Resolved(uint256 indexed id, Outcome outcome);
    event Claimed(uint256 indexed id, address indexed user, uint128 amount);
    event FeeChanged(uint256 prev, uint256 next);

    error BadParams();
    error NotOpen();
    error NotEnded();
    error AlreadyResolved();
    error NotResolver();
    error BadOutcome();
    error NoStake();
    error EthSendFail();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function create(address resolver, uint64 endsAt, bytes32 question, string calldata detailsCid)
        external returns (uint256 id)
    {
        if (resolver == address(0) || endsAt <= block.timestamp) revert BadParams();
        _payFee(createFeeBvm);
        id = markets.length;
        markets.push();
        Market storage m = markets[id];
        m.creator = msg.sender; m.resolver = resolver; m.endsAt = endsAt;
        m.question = question; m.detailsCid = detailsCid;
        emit Created(id, msg.sender, resolver, endsAt, question, detailsCid);
    }

    function bet(uint256 id, bool yes) external payable {
        Market storage m = markets[id];
        if (m.outcome != Outcome.Pending || block.timestamp > m.endsAt) revert NotOpen();
        if (msg.value == 0) revert BadParams();
        if (yes) { yesBets[id][msg.sender] += uint128(msg.value); m.yesPool += uint128(msg.value); }
        else     { noBets[id][msg.sender]  += uint128(msg.value); m.noPool  += uint128(msg.value); }
        emit Bet(id, msg.sender, yes, uint128(msg.value));
    }

    function resolve(uint256 id, Outcome outcome) external {
        Market storage m = markets[id];
        if (msg.sender != m.resolver) revert NotResolver();
        if (m.outcome != Outcome.Pending) revert AlreadyResolved();
        if (block.timestamp <= m.endsAt) revert NotEnded();
        if (outcome == Outcome.Pending) revert BadOutcome();
        m.outcome = outcome;
        emit Resolved(id, outcome);
    }

    function claim(uint256 id) external {
        Market storage m = markets[id];
        if (m.outcome == Outcome.Pending) revert NotEnded();
        uint128 payout;
        if (m.outcome == Outcome.Void) {
            payout = yesBets[id][msg.sender] + noBets[id][msg.sender];
            yesBets[id][msg.sender] = 0;
            noBets[id][msg.sender]  = 0;
        } else if (m.outcome == Outcome.Yes) {
            uint128 stake = yesBets[id][msg.sender];
            if (stake == 0) revert NoStake();
            yesBets[id][msg.sender] = 0;
            payout = uint128((uint256(stake) * (uint256(m.yesPool) + uint256(m.noPool))) / uint256(m.yesPool));
        } else {
            uint128 stake = noBets[id][msg.sender];
            if (stake == 0) revert NoStake();
            noBets[id][msg.sender] = 0;
            payout = uint128((uint256(stake) * (uint256(m.yesPool) + uint256(m.noPool))) / uint256(m.noPool));
        }
        if (payout == 0) revert NoStake();
        (bool ok, ) = msg.sender.call{value: payout}("");
        if (!ok) revert EthSendFail();
        emit Claimed(id, msg.sender, payout);
    }

    function totalMarkets() external view returns (uint256) { return markets.length; }

    function setFee(uint256 next) external onlyOwner {
        emit FeeChanged(createFeeBvm, next);
        createFeeBvm = next;
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
