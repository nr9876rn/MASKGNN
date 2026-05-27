// ===== FILE: contracts_-_bvm/utilities/BVMSplitter.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

interface IERC20Min { function transfer(address, uint256) external returns (bool); function transferFrom(address, address, uint256) external returns (bool); function balanceOf(address) external view returns (uint256); }

contract BVMSplitter is BVMFeeBase {
    uint256 public setupFeeBvm = 210 ether;

    struct Split {
        address creator;
        address[] payees;
        uint32[]  shares;
        uint32    totalShares;
        uint128   totalEthReleased;
        mapping(address => uint128) tokenReleased;
        mapping(address => mapping(address => uint128)) payeeTokenClaimed;
        mapping(address => uint128) payeeEthClaimed;
    }
    mapping(uint256 => Split) private _splits;
    uint256 public nextSplitId = 1;
    mapping(address => uint256[]) public splitsOf;

    event SplitCreated(uint256 indexed id, address indexed creator, address[] payees, uint32[] shares);
    event EthReleased(uint256 indexed id, address indexed payee, uint128 amount);
    event TokenReleased(uint256 indexed id, address indexed token, address indexed payee, uint128 amount);
    event Funded(uint256 indexed id, address indexed funder, uint128 amount);
    event FeeChanged(uint256 prev, uint256 next);

    error BadParams();
    error NoSplit();
    error NotPayee();
    error Empty();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function create(address[] calldata payees, uint32[] calldata shares) external returns (uint256 id) {
        if (payees.length == 0 || payees.length != shares.length || payees.length > 64) revert BadParams();
        uint32 total;
        for (uint256 i; i < shares.length; ) {
            if (payees[i] == address(0) || shares[i] == 0) revert BadParams();
            total += shares[i];
            unchecked { i++; }
        }
        if (total == 0) revert BadParams();
        _payFee(setupFeeBvm);

        id = nextSplitId++;
        Split storage s = _splits[id];
        s.creator = msg.sender;
        s.payees = payees;
        s.shares = shares;
        s.totalShares = total;
        for (uint256 i; i < payees.length; ) {
            splitsOf[payees[i]].push(id);
            unchecked { i++; }
        }
        emit SplitCreated(id, msg.sender, payees, shares);
    }

    receive() external payable {}

    function fundEth(uint256 id) external payable {
        if (_splits[id].totalShares == 0) revert NoSplit();
        emit Funded(id, msg.sender, uint128(msg.value));
    }

    function releaseEth(uint256 id, address payee) external {
        Split storage s = _splits[id];
        if (s.totalShares == 0) revert NoSplit();
        uint32 sh = _shareOf(s, payee);
        if (sh == 0) revert NotPayee();
        uint128 total = uint128(address(this).balance + uint256(s.totalEthReleased));
        uint128 due   = uint128((uint256(total) * uint256(sh)) / uint256(s.totalShares)) - s.payeeEthClaimed[payee];
        if (due == 0) revert Empty();
        s.payeeEthClaimed[payee] += due;
        s.totalEthReleased       += due;
        (bool ok, ) = payee.call{value: due}("");
        require(ok, "eth");
        emit EthReleased(id, payee, due);
    }

    function releaseToken(uint256 id, address token, address payee) external {
        Split storage s = _splits[id];
        if (s.totalShares == 0) revert NoSplit();
        uint32 sh = _shareOf(s, payee);
        if (sh == 0) revert NotPayee();
        uint128 totalNow = uint128(IERC20Min(token).balanceOf(address(this)) + uint256(s.tokenReleased[token]));
        uint128 due      = uint128((uint256(totalNow) * uint256(sh)) / uint256(s.totalShares)) - s.payeeTokenClaimed[token][payee];
        if (due == 0) revert Empty();
        s.payeeTokenClaimed[token][payee] += due;
        s.tokenReleased[token]            += due;
        require(IERC20Min(token).transfer(payee, due), "tk");
        emit TokenReleased(id, token, payee, due);
    }

    function viewSplit(uint256 id) external view returns (address creator, address[] memory payees, uint32[] memory shares, uint32 totalShares) {
        Split storage s = _splits[id];
        return (s.creator, s.payees, s.shares, s.totalShares);
    }

    function shareOf(uint256 id, address payee) external view returns (uint32) {
        return _shareOf(_splits[id], payee);
    }

    function _shareOf(Split storage s, address payee) internal view returns (uint32) {
        for (uint256 i; i < s.payees.length; ) {
            if (s.payees[i] == payee) return s.shares[i];
            unchecked { i++; }
        }
        return 0;
    }

    function setFee(uint256 next) external onlyOwner {
        emit FeeChanged(setupFeeBvm, next);
        setupFeeBvm = next;
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
