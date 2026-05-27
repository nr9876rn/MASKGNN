// ===== FILE: contracts_-_bvm/BVMNodes.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

interface IERC721Receiver {
    function onERC721Received(address, address, uint256, bytes calldata) external returns (bytes4);
}

interface ITreasury {
    function payOut(address to, uint256 amount) external returns (uint256 paid);
    function reserveLeft() external view returns (uint256);
}

contract BVMNodes {
    string public constant name   = "BitcoinVM Nodes";
    string public constant symbol = "BVMN";

    uint8  public constant TIER_COUNT      = 5;
    uint256 public constant EPOCH_DURATION = 182 days;
    uint256 public constant ACC_SCALE      = 1e18;

    uint256 public constant INITIAL_RATE_PER_SEC = 333_900_000_000_000_000;

    struct Tier {
        uint128 priceWei;
        uint32  multiplier;
        uint32  maxSupply;
        uint32  minted;
    }
    Tier[5] public _tiers;

    struct Node {
        uint8   tier;
        uint64  mintedAt;
        uint32  units;
        uint256 rewardDebt;
    }
    mapping(uint256 => Node) public nodes;

    mapping(uint256 => address) public ownerOf;
    mapping(address => uint256) public balanceOf;
    mapping(uint256 => address) public getApproved;
    mapping(address => mapping(address => bool)) public isApprovedForAll;
    mapping(address => uint256) public unitsOf;

    ITreasury public immutable treasury;
    address   public treasuryAddr;
    address   public ethTreasury;
    address   public owner;
    string    private _metaRoot;

    uint64  public genesis;
    bool    public rewardsEnabled;
    bool    public mintOpen;

    mapping(address => bool) public whitelisted;

    uint256 public totalUnits;
    uint256 public accBvmPerUnit;
    uint64  public lastUpdate;

    uint256 public totalMinted;
    uint256 public totalDistributed;
    uint256 public totalTributeIn;

    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);
    event Approval(address indexed owner, address indexed approved, uint256 indexed tokenId);
    event ApprovalForAll(address indexed owner, address indexed operator, bool approved);
    event Forged(uint256 indexed tokenId, address indexed who, uint8 tier, uint256 priceWei);
    event Harvested(address indexed who, uint256 totalDue, uint256 totalPaid, uint256 nodes_);
    event Tribute(uint256 amount, uint256 newAccPerUnit);
    event MintOpened(uint64 atTimestamp);
    event RewardsEnabled(uint64 atTimestamp);
    event WhitelistSet(address indexed who, bool flag);
    event EthTreasurySet(address indexed prev, address indexed next);
    event MetaRootSet(string root);
    event OwnershipTransferred(address indexed prev, address indexed next);

    error NotOwner();
    error NotTreasury();
    error MintNotOpen();
    error MintAlreadyOpen();
    error RewardsAlreadyEnabled();
    error RewardsNotEnabled();
    error BadTier();
    error SoldOut();
    error WrongPrice();
    error TokenMissing();
    error NotTokenOwner();
    error NotAuthorized();
    error ZeroAddress();
    error UnsafeReceiver();
    error EmptyList();
    error EthSendFail();

    modifier onlyOwner()    { if (msg.sender != owner) revert NotOwner(); _; }
    modifier onlyTreasury() { if (msg.sender != treasuryAddr) revert NotTreasury(); _; }
    modifier mintAllowed()  { if (!mintOpen && !whitelisted[msg.sender]) revert MintNotOpen(); _; }

    constructor(address _treasury, address _owner, address _ethTreasury) {
        if (_treasury == address(0) || _owner == address(0) || _ethTreasury == address(0)) revert ZeroAddress();
        treasury     = ITreasury(_treasury);
        treasuryAddr = _treasury;
        owner        = _owner;
        ethTreasury  = _ethTreasury;

        _tiers[0] = Tier({ priceWei: 0.005 ether, multiplier:   1, maxSupply: 21_000, minted: 0 });
        _tiers[1] = Tier({ priceWei: 0.025 ether, multiplier:   4, maxSupply:  8_400, minted: 0 });
        _tiers[2] = Tier({ priceWei: 0.100 ether, multiplier:  16, maxSupply:  2_100, minted: 0 });
        _tiers[3] = Tier({ priceWei: 0.500 ether, multiplier:  64, maxSupply:    420, minted: 0 });
        _tiers[4] = Tier({ priceWei: 2.500 ether, multiplier: 256, maxSupply:     84, minted: 0 });
    }

    function tiers(uint8 t) external view returns (uint128 priceWei, uint32 multiplier, uint32 maxSupply, uint32 minted) {
        Tier memory x = _tiers[t];
        return (x.priceWei, x.multiplier, x.maxSupply, x.minted);
    }

    function openMint() external onlyOwner {
        if (mintOpen) revert MintAlreadyOpen();
        mintOpen = true;
        emit MintOpened(uint64(block.timestamp));
    }

    function enableRewards() external onlyOwner {
        if (rewardsEnabled) revert RewardsAlreadyEnabled();
        rewardsEnabled = true;
        genesis = uint64(block.timestamp);
        lastUpdate = uint64(block.timestamp);
        emit RewardsEnabled(genesis);
    }

    function setWhitelisted(address a, bool flag) external onlyOwner {
        if (a == address(0)) revert ZeroAddress();
        whitelisted[a] = flag;
        emit WhitelistSet(a, flag);
    }

    function setWhitelistedMany(address[] calldata addrs, bool flag) external onlyOwner {
        for (uint256 i; i < addrs.length; ) {
            address a = addrs[i];
            if (a != address(0)) {
                whitelisted[a] = flag;
                emit WhitelistSet(a, flag);
            }
            unchecked { i++; }
        }
    }

    function era() public view returns (uint256) {
        if (!rewardsEnabled) return 0;
        return (block.timestamp - uint256(genesis)) / EPOCH_DURATION;
    }

    function rateAtEra(uint256 e) public pure returns (uint256) {
        if (e >= 32) return 0;
        return INITIAL_RATE_PER_SEC >> e;
    }

    function currentRate() external view returns (uint256) {
        return rewardsEnabled ? rateAtEra(era()) : 0;
    }

    function nextHalvingIn() external view returns (uint256) {
        if (!rewardsEnabled) return EPOCH_DURATION;
        uint256 elapsed = (block.timestamp - uint256(genesis)) % EPOCH_DURATION;
        return EPOCH_DURATION - elapsed;
    }

    function reserveLeft() external view returns (uint256) {
        return treasury.reserveLeft();
    }

    function _emissionBetween(uint64 fromT, uint64 toT) internal view returns (uint256 total) {
        if (toT <= fromT || !rewardsEnabled) return 0;
        uint256 cursor = uint256(fromT);
        uint256 endT   = uint256(toT);
        while (cursor < endT) {
            uint256 eIdx;
            unchecked { eIdx = (cursor - uint256(genesis)) / EPOCH_DURATION; }
            uint256 rate = rateAtEra(eIdx);
            if (rate == 0) break;
            uint256 nextBoundary = uint256(genesis) + (eIdx + 1) * EPOCH_DURATION;
            uint256 segEnd = endT < nextBoundary ? endT : nextBoundary;
            total += rate * (segEnd - cursor);
            cursor = segEnd;
        }
    }

    function _updatePool() internal {
        if (!rewardsEnabled) { lastUpdate = uint64(block.timestamp); return; }
        if (totalUnits == 0) { lastUpdate = uint64(block.timestamp); return; }
        uint256 emitted = _emissionBetween(lastUpdate, uint64(block.timestamp));
        uint256 reserve = treasury.reserveLeft();
        if (emitted > reserve) emitted = reserve;
        if (emitted > 0) {
            accBvmPerUnit += (emitted * ACC_SCALE) / totalUnits;
        }
        lastUpdate = uint64(block.timestamp);
    }

    function applyTribute(uint256 amount) external onlyTreasury {
        if (amount == 0 || totalUnits == 0) return;
        accBvmPerUnit += (amount * ACC_SCALE) / totalUnits;
        unchecked { totalTributeIn += amount; }
        emit Tribute(amount, accBvmPerUnit);
    }

    function pendingOf(uint256 tokenId) public view returns (uint256) {
        if (ownerOf[tokenId] == address(0)) return 0;
        Node memory n = nodes[tokenId];
        uint256 acc = accBvmPerUnit;
        if (rewardsEnabled && totalUnits > 0) {
            uint256 emitted = _emissionBetween(lastUpdate, uint64(block.timestamp));
            uint256 reserve = treasury.reserveLeft();
            if (emitted > reserve) emitted = reserve;
            if (emitted > 0) acc += (emitted * ACC_SCALE) / totalUnits;
        }
        uint256 owed = (uint256(n.units) * acc) / ACC_SCALE;
        if (owed <= n.rewardDebt) return 0;
        return owed - n.rewardDebt;
    }

    function pendingForOwner(address who, uint256[] calldata ids) external view returns (uint256 total) {
        for (uint256 i; i < ids.length; ) {
            uint256 id = ids[i];
            if (ownerOf[id] == who) total += pendingOf(id);
            unchecked { i++; }
        }
    }

    function forge(uint8 tier) external payable mintAllowed returns (uint256 tokenId) {
        if (tier >= TIER_COUNT) revert BadTier();
        Tier storage t = _tiers[tier];
        if (t.minted >= t.maxSupply) revert SoldOut();
        if (msg.value != uint256(t.priceWei)) revert WrongPrice();

        _updatePool();

        unchecked { tokenId = ++totalMinted; t.minted += 1; }
        uint32 units = t.multiplier;
        nodes[tokenId] = Node({
            tier:       tier,
            mintedAt:   uint64(block.timestamp),
            units:      units,
            rewardDebt: (uint256(units) * accBvmPerUnit) / ACC_SCALE
        });
        ownerOf[tokenId] = msg.sender;
        unchecked {
            balanceOf[msg.sender] += 1;
            unitsOf[msg.sender]   += units;
            totalUnits            += units;
        }

        (bool ok, ) = ethTreasury.call{value: msg.value}("");
        if (!ok) revert EthSendFail();

        emit Transfer(address(0), msg.sender, tokenId);
        emit Forged(tokenId, msg.sender, tier, msg.value);
    }

    function forgeMany(uint8 tier, uint256 count) external payable mintAllowed returns (uint256 firstId) {
        if (tier >= TIER_COUNT) revert BadTier();
        if (count == 0) revert EmptyList();
        Tier storage t = _tiers[tier];
        if (uint256(t.minted) + count > uint256(t.maxSupply)) revert SoldOut();
        uint256 unit = uint256(t.priceWei);
        if (msg.value != unit * count) revert WrongPrice();

        _updatePool();

        firstId = totalMinted + 1;
        uint32 units = t.multiplier;
        uint256 unitsAddedTotal;
        uint256 debt = (uint256(units) * accBvmPerUnit) / ACC_SCALE;
        for (uint256 i; i < count; ) {
            unchecked { totalMinted += 1; }
            uint256 tokenId = totalMinted;
            nodes[tokenId] = Node({
                tier:       tier,
                mintedAt:   uint64(block.timestamp),
                units:      units,
                rewardDebt: debt
            });
            ownerOf[tokenId] = msg.sender;
            emit Transfer(address(0), msg.sender, tokenId);
            emit Forged(tokenId, msg.sender, tier, unit);
            unchecked { unitsAddedTotal += units; i++; }
        }
        unchecked {
            balanceOf[msg.sender] += count;
            unitsOf[msg.sender]   += unitsAddedTotal;
            totalUnits            += unitsAddedTotal;
            t.minted              += uint32(count);
        }

        (bool ok, ) = ethTreasury.call{value: msg.value}("");
        if (!ok) revert EthSendFail();
    }

    function harvest(uint256[] calldata ids) external returns (uint256 paid) {
        if (ids.length == 0) revert EmptyList();
        if (!rewardsEnabled) revert RewardsNotEnabled();
        _updatePool();
        uint256 due;
        for (uint256 i; i < ids.length; ) {
            uint256 id = ids[i];
            if (ownerOf[id] != msg.sender) revert NotTokenOwner();
            Node storage n = nodes[id];
            uint256 owed = (uint256(n.units) * accBvmPerUnit) / ACC_SCALE;
            if (owed > n.rewardDebt) {
                due += owed - n.rewardDebt;
                n.rewardDebt = owed;
            }
            unchecked { i++; }
        }
        if (due == 0) {
            emit Harvested(msg.sender, 0, 0, ids.length);
            return 0;
        }
        paid = treasury.payOut(msg.sender, due);
        unchecked { totalDistributed += paid; }
        emit Harvested(msg.sender, due, paid, ids.length);
    }

    function approve(address to, uint256 id) external {
        address o = ownerOf[id];
        if (o == address(0)) revert TokenMissing();
        if (msg.sender != o && !isApprovedForAll[o][msg.sender]) revert NotAuthorized();
        getApproved[id] = to;
        emit Approval(o, to, id);
    }

    function setApprovalForAll(address op, bool ok) external {
        isApprovedForAll[msg.sender][op] = ok;
        emit ApprovalForAll(msg.sender, op, ok);
    }

    function transferFrom(address from, address to, uint256 id) public {
        if (to == address(0)) revert ZeroAddress();
        if (ownerOf[id] != from) revert NotTokenOwner();
        if (msg.sender != from
            && getApproved[id] != msg.sender
            && !isApprovedForAll[from][msg.sender]) revert NotAuthorized();

        _updatePool();
        Node storage n = nodes[id];
        uint256 owed = (uint256(n.units) * accBvmPerUnit) / ACC_SCALE;
        if (owed > n.rewardDebt) {
            uint256 due = owed - n.rewardDebt;
            n.rewardDebt = owed;
            if (due > 0) {
                uint256 paid = treasury.payOut(from, due);
                unchecked { totalDistributed += paid; }
                emit Harvested(from, due, paid, 1);
            }
        }
        n.rewardDebt = (uint256(n.units) * accBvmPerUnit) / ACC_SCALE;

        unchecked {
            balanceOf[from] -= 1;
            balanceOf[to]   += 1;
            unitsOf[from]   -= n.units;
            unitsOf[to]     += n.units;
        }
        ownerOf[id] = to;
        delete getApproved[id];
        emit Transfer(from, to, id);
    }

    function safeTransferFrom(address from, address to, uint256 id) external {
        safeTransferFrom(from, to, id, "");
    }

    function safeTransferFrom(address from, address to, uint256 id, bytes memory data) public {
        transferFrom(from, to, id);
        if (to.code.length != 0) {
            try IERC721Receiver(to).onERC721Received(msg.sender, from, id, data) returns (bytes4 r) {
                if (r != IERC721Receiver.onERC721Received.selector) revert UnsafeReceiver();
            } catch { revert UnsafeReceiver(); }
        }
    }

    function tokenURI(uint256 id) external view returns (string memory) {
        if (ownerOf[id] == address(0)) revert TokenMissing();
        return string.concat(_metaRoot, _itoa(id), ".json");
    }

    function metaRoot() external view returns (string memory) { return _metaRoot; }

    function supportsInterface(bytes4 i) external pure returns (bool) {
        return i == 0x80ac58cd || i == 0x5b5e139f || i == 0x01ffc9a7;
    }

    function setMetaRoot(string calldata root) external onlyOwner {
        _metaRoot = root;
        emit MetaRootSet(root);
    }

    function setEthTreasury(address t) external onlyOwner {
        if (t == address(0)) revert ZeroAddress();
        emit EthTreasurySet(ethTreasury, t);
        ethTreasury = t;
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

    function _itoa(uint256 v) internal pure returns (string memory) {
        if (v == 0) return "0";
        uint256 t = v; uint256 d;
        while (t != 0) { d++; t /= 10; }
        bytes memory s = new bytes(d);
        while (v != 0) { d--; s[d] = bytes1(uint8(48 + v % 10)); v /= 10; }
        return string(s);
    }
}
