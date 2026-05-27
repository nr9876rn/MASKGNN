// ===== FILE: contracts_-_bvm/utilities/BVMTipJar.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

interface INames {
    function addressOfName(string calldata) external view returns (address);
}

contract BVMTipJar is BVMFeeBase {
    uint256 public tipFeeBvm = 210 ether;
    INames  public names;

    struct Tip {
        address from;
        address to;
        uint64  timestamp;
        uint256 amountBvm;
    }
    Tip[] public tips;
    mapping(address => uint256) public tipsReceivedCount;
    mapping(address => uint256) public tipsReceivedBvm;
    mapping(address => uint256) public tipsSentCount;
    mapping(address => uint256) public tipsSentBvm;

    event Tipped(address indexed from, address indexed to, uint256 amountBvm, uint256 feeBvm, uint256 indexed tipId);
    event FeeChanged(uint256 prev, uint256 next);
    event NamesBound(address indexed prev, address indexed next);

    error NoTarget();
    error InvalidTip();
    error TransferFailed();

    constructor(address _treasury, address _owner, address _names) BVMFeeBase(_treasury, _owner) {
        if (_names != address(0)) names = INames(_names);
    }

    function tipAddress(address to, uint256 amount) external returns (uint256 id) {
        if (to == address(0)) revert NoTarget();
        if (amount == 0) revert InvalidTip();
        _payFee(tipFeeBvm);
        _doTip(msg.sender, to, amount);
        id = tips.length - 1;
    }

    function tipName(string calldata n, uint256 amount) external returns (uint256 id) {
        if (address(names) == address(0)) revert NoTarget();
        address to = names.addressOfName(n);
        if (to == address(0)) revert NoTarget();
        if (amount == 0) revert InvalidTip();
        _payFee(tipFeeBvm);
        _doTip(msg.sender, to, amount);
        id = tips.length - 1;
    }

    function _doTip(address from, address to, uint256 amount) internal {
        IBVM bvm = IBVM(address(_tokenAddr()));
        if (!bvm.transferFrom(from, to, amount)) revert TransferFailed();
        tips.push(Tip({ from: from, to: to, timestamp: uint64(block.timestamp), amountBvm: amount }));
        unchecked {
            tipsReceivedCount[to]++; tipsReceivedBvm[to] += amount;
            tipsSentCount[from]++;   tipsSentBvm[from]   += amount;
        }
        emit Tipped(from, to, amount, tipFeeBvm, tips.length - 1);
    }

    function _tokenAddr() internal view returns (address) {
        // The BVM token is the same one the treasury holds. We get it via a static call.
        // (treasury exposes `token()`.)
        (bool ok, bytes memory data) = address(bvmTreasury).staticcall(abi.encodeWithSignature("token()"));
        require(ok && data.length == 32, "token");
        return abi.decode(data, (address));
    }

    function totalTips() external view returns (uint256) { return tips.length; }

    function setFee(uint256 next) external onlyOwner {
        emit FeeChanged(tipFeeBvm, next);
        tipFeeBvm = next;
    }

    function setNames(address n) external onlyOwner {
        emit NamesBound(address(names), n);
        names = INames(n);
    }
}

interface IBVM {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
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
