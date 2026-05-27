// ===== FILE: contracts_-_bvm/utilities/BVMMerkleDrop.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

interface IERC20Min { function transfer(address, uint256) external returns (bool); function transferFrom(address, address, uint256) external returns (bool); }

contract BVMMerkleDrop is BVMFeeBase {
    uint256 public createFeeBvm = 500 ether;

    struct Campaign {
        address creator;
        address token;
        bytes32 root;
        uint128 totalAllocated;
        uint128 totalClaimed;
        uint64  closesAt;
        bool    refunded;
    }
    Campaign[] public campaigns;
    mapping(uint256 => mapping(uint256 => uint256)) private _claimedBitmap;

    event CampaignCreated(uint256 indexed id, address indexed creator, address indexed token, bytes32 root, uint128 total, uint64 closesAt);
    event Claimed(uint256 indexed id, uint256 indexed index, address indexed account, uint128 amount);
    event Refunded(uint256 indexed id, uint128 amount);
    event FeeChanged(uint256 prev, uint256 next);

    error BadParams();
    error InvalidProof();
    error AlreadyClaimed();
    error CampaignClosed();
    error NotCreator();
    error NotClosed();
    error AlreadyRefunded();
    error TransferFailed();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function createCampaign(address token, bytes32 root, uint128 total, uint64 closesAt) external returns (uint256 id) {
        if (token == address(0) || root == bytes32(0) || total == 0 || closesAt <= block.timestamp) revert BadParams();
        _payFee(createFeeBvm);
        if (!IERC20Min(token).transferFrom(msg.sender, address(this), total)) revert TransferFailed();
        id = campaigns.length;
        campaigns.push(Campaign({
            creator: msg.sender, token: token, root: root,
            totalAllocated: total, totalClaimed: 0,
            closesAt: closesAt, refunded: false
        }));
        emit CampaignCreated(id, msg.sender, token, root, total, closesAt);
    }

    function isClaimed(uint256 id, uint256 index) public view returns (bool) {
        uint256 word = index / 256;
        uint256 bit  = index % 256;
        return _claimedBitmap[id][word] & (1 << bit) != 0;
    }

    function claim(uint256 id, uint256 index, address account, uint128 amount, bytes32[] calldata proof) external {
        Campaign storage c = campaigns[id];
        if (block.timestamp > c.closesAt) revert CampaignClosed();
        if (isClaimed(id, index)) revert AlreadyClaimed();
        bytes32 leaf = keccak256(abi.encodePacked(index, account, amount));
        bytes32 node = leaf;
        for (uint256 i; i < proof.length; ) {
            bytes32 p = proof[i];
            node = node <= p ? keccak256(abi.encodePacked(node, p)) : keccak256(abi.encodePacked(p, node));
            unchecked { i++; }
        }
        if (node != c.root) revert InvalidProof();
        uint256 word = index / 256;
        uint256 bit  = index % 256;
        _claimedBitmap[id][word] |= (1 << bit);
        c.totalClaimed += amount;
        if (!IERC20Min(c.token).transfer(account, amount)) revert TransferFailed();
        emit Claimed(id, index, account, amount);
    }

    function refund(uint256 id) external {
        Campaign storage c = campaigns[id];
        if (msg.sender != c.creator) revert NotCreator();
        if (block.timestamp <= c.closesAt) revert NotClosed();
        if (c.refunded) revert AlreadyRefunded();
        c.refunded = true;
        uint128 left = c.totalAllocated - c.totalClaimed;
        if (left > 0) {
            if (!IERC20Min(c.token).transfer(c.creator, left)) revert TransferFailed();
        }
        emit Refunded(id, left);
    }

    function totalCampaigns() external view returns (uint256) { return campaigns.length; }

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
