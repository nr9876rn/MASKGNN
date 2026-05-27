// ===== FILE: contracts_-_bvm/utilities/BVMBounty.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

interface IERC20Min { function transfer(address, uint256) external returns (bool); function transferFrom(address, address, uint256) external returns (bool); }

contract BVMBounty is BVMFeeBase {
    uint256 public postFeeBvm = 210 ether;

    enum Status { Open, Awarded, Cancelled }

    struct Bounty {
        address poster;
        address rewardToken;
        uint128 rewardAmount;
        uint64  deadline;
        Status  status;
        bytes32 brief;
        string  detailsCid;
        address winner;
        string  submissionCid;
    }
    Bounty[] public bounties;
    mapping(address => uint256[]) public bountiesOf;

    event Posted(uint256 indexed id, address indexed poster, address indexed token, uint128 amount, uint64 deadline, bytes32 brief, string detailsCid);
    event Submitted(uint256 indexed id, address indexed submitter, string submissionCid);
    event Awarded(uint256 indexed id, address indexed winner, uint128 amount);
    event Cancelled(uint256 indexed id);
    event FeeChanged(uint256 prev, uint256 next);

    error BadParams();
    error NotPoster();
    error NotOpen();
    error TransferFailed();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function post(address rewardToken, uint128 rewardAmount, uint64 deadline, bytes32 brief, string calldata detailsCid)
        external returns (uint256 id)
    {
        if (rewardToken == address(0) || rewardAmount == 0 || deadline <= block.timestamp) revert BadParams();
        _payFee(postFeeBvm);
        if (!IERC20Min(rewardToken).transferFrom(msg.sender, address(this), rewardAmount)) revert TransferFailed();
        id = bounties.length;
        bounties.push();
        Bounty storage b = bounties[id];
        b.poster = msg.sender; b.rewardToken = rewardToken; b.rewardAmount = rewardAmount;
        b.deadline = deadline; b.brief = brief; b.detailsCid = detailsCid;
        bountiesOf[msg.sender].push(id);
        emit Posted(id, msg.sender, rewardToken, rewardAmount, deadline, brief, detailsCid);
    }

    function submit(uint256 id, string calldata submissionCid) external {
        if (bounties[id].status != Status.Open) revert NotOpen();
        emit Submitted(id, msg.sender, submissionCid);
    }

    function award(uint256 id, address winner, string calldata submissionCid) external {
        Bounty storage b = bounties[id];
        if (msg.sender != b.poster) revert NotPoster();
        if (b.status != Status.Open) revert NotOpen();
        b.status = Status.Awarded;
        b.winner = winner;
        b.submissionCid = submissionCid;
        if (!IERC20Min(b.rewardToken).transfer(winner, b.rewardAmount)) revert TransferFailed();
        emit Awarded(id, winner, b.rewardAmount);
    }

    function cancel(uint256 id) external {
        Bounty storage b = bounties[id];
        if (msg.sender != b.poster) revert NotPoster();
        if (b.status != Status.Open) revert NotOpen();
        if (block.timestamp <= b.deadline) revert NotOpen();
        b.status = Status.Cancelled;
        if (!IERC20Min(b.rewardToken).transfer(b.poster, b.rewardAmount)) revert TransferFailed();
        emit Cancelled(id);
    }

    function totalBounties() external view returns (uint256) { return bounties.length; }

    function setFee(uint256 next) external onlyOwner {
        emit FeeChanged(postFeeBvm, next);
        postFeeBvm = next;
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
