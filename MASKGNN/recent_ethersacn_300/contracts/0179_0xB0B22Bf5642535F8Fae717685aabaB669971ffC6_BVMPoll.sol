// ===== FILE: contracts_-_bvm/utilities/BVMPoll.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

interface IBVMBal { function balanceOf(address) external view returns (uint256); }

contract BVMPoll is BVMFeeBase {
    uint256 public createFeeBvm = 210 ether;

    struct Poll {
        address creator;
        uint64  endsAt;
        bytes32 question;
        string  detailsCid;
        uint8   optionCount;
    }
    Poll[] public polls;
    mapping(uint256 => mapping(uint8 => uint256)) public votes;        // pollId → option → weight
    mapping(uint256 => mapping(address => bool))  public hasVoted;
    mapping(address => uint256[]) public pollsBy;

    IBVMBal public immutable bvm;

    event PollCreated(uint256 indexed id, address indexed creator, uint8 optionCount, uint64 endsAt, bytes32 question, string detailsCid);
    event Voted(uint256 indexed id, address indexed voter, uint8 option, uint256 weight);
    event FeeChanged(uint256 prev, uint256 next);

    error BadParams();
    error PollOver();
    error AlreadyVoted();
    error NoWeight();
    error BadOption();

    constructor(address _treasury, address _owner, address _bvm) BVMFeeBase(_treasury, _owner) {
        if (_bvm == address(0)) revert ZeroAddress();
        bvm = IBVMBal(_bvm);
    }

    function createPoll(uint8 optionCount, uint64 endsAt, bytes32 question, string calldata detailsCid)
        external returns (uint256 id)
    {
        if (optionCount < 2 || optionCount > 16) revert BadParams();
        if (endsAt <= block.timestamp) revert BadParams();
        _payFee(createFeeBvm);
        id = polls.length;
        polls.push(Poll({
            creator: msg.sender, endsAt: endsAt,
            question: question, detailsCid: detailsCid,
            optionCount: optionCount
        }));
        pollsBy[msg.sender].push(id);
        emit PollCreated(id, msg.sender, optionCount, endsAt, question, detailsCid);
    }

    function vote(uint256 id, uint8 option) external {
        Poll storage p = polls[id];
        if (block.timestamp > p.endsAt) revert PollOver();
        if (hasVoted[id][msg.sender]) revert AlreadyVoted();
        if (option >= p.optionCount) revert BadOption();
        uint256 w = bvm.balanceOf(msg.sender);
        if (w == 0) revert NoWeight();
        hasVoted[id][msg.sender] = true;
        votes[id][option] += w;
        emit Voted(id, msg.sender, option, w);
    }

    function totalPolls() external view returns (uint256) { return polls.length; }

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
