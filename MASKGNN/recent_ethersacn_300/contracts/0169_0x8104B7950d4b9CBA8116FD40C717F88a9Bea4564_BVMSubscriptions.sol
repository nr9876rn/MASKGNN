// ===== FILE: contracts_-_bvm/utilities/BVMSubscriptions.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

interface IERC20Min { function transferFrom(address, address, uint256) external returns (bool); }

contract BVMSubscriptions is BVMFeeBase {
    uint256 public setupFeeBvm = 210 ether;
    uint256 public chargeFeeBvm = 50 ether;

    struct Sub {
        address payer;
        address payee;
        address token;
        uint128 amountPerCharge;
        uint64  intervalSecs;
        uint64  nextCharge;
        bool    active;
    }
    Sub[] public subs;
    mapping(address => uint256[]) public subsBy;
    mapping(address => uint256[]) public subsTo;

    event SubCreated(uint256 indexed id, address indexed payer, address indexed payee, address token, uint128 perCharge, uint64 interval, uint64 firstChargeAt);
    event Charged(uint256 indexed id, uint128 amount, uint64 nextChargeAt);
    event Cancelled(uint256 indexed id);
    event FeeChanged(uint256 prev, uint256 setup, uint256 perCharge);

    error BadParams();
    error NotPayer();
    error Inactive();
    error TooEarly();
    error TransferFailed();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function create(address payee, address token, uint128 perCharge, uint64 interval, uint64 firstChargeAt)
        external returns (uint256 id)
    {
        if (payee == address(0) || token == address(0) || perCharge == 0 || interval == 0 || firstChargeAt < block.timestamp) revert BadParams();
        _payFee(setupFeeBvm);
        id = subs.length;
        subs.push(Sub({
            payer: msg.sender, payee: payee, token: token,
            amountPerCharge: perCharge, intervalSecs: interval,
            nextCharge: firstChargeAt, active: true
        }));
        subsBy[msg.sender].push(id);
        subsTo[payee].push(id);
        emit SubCreated(id, msg.sender, payee, token, perCharge, interval, firstChargeAt);
    }

    function charge(uint256 id) external {
        Sub storage s = subs[id];
        if (!s.active) revert Inactive();
        if (block.timestamp < s.nextCharge) revert TooEarly();
        _payFee(chargeFeeBvm);
        if (!IERC20Min(s.token).transferFrom(s.payer, s.payee, s.amountPerCharge)) revert TransferFailed();
        s.nextCharge = uint64(block.timestamp) + s.intervalSecs;
        emit Charged(id, s.amountPerCharge, s.nextCharge);
    }

    function cancel(uint256 id) external {
        Sub storage s = subs[id];
        if (msg.sender != s.payer) revert NotPayer();
        if (!s.active) revert Inactive();
        s.active = false;
        emit Cancelled(id);
    }

    function totalSubs() external view returns (uint256) { return subs.length; }

    function setFees(uint256 _setup, uint256 _perCharge) external onlyOwner {
        setupFeeBvm = _setup; chargeFeeBvm = _perCharge;
        emit FeeChanged(0, _setup, _perCharge);
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
