// ===== FILE: src/XMRP2P.sol =====
// SPDX-License-Identifier: MIT
//
// ==============================
// xmrp2p.eth - v1
//
// Published by v3xlabs
// ==============================

pragma solidity ^0.8.34;

import {Ed25519} from "./Ed25519.sol";
import "./Errors.sol";
import "./Enums.sol";
import {Ownable} from "solady/auth/Ownable.sol";

contract XMRP2P is Ownable {
    struct Offer {
        uint256 id;
        OfferType kind;
        OfferState state;
        address owner;
        address counterparty;
        uint256 amount;
        uint256 deposit;
        uint256 xmrAmount;
        uint256 lastupdate;
        uint256 blockTaken;
        /// Monero public spend key of the EVM side of the trade
        uint256 evmPublicSpendKey;
        /// Monero private spend key of the EVM side of the trade. Set during a call to refund.
        uint256 evmPrivateSpendKey;
        /// Public view key provided by the EVM side of the trade. This is needed to compute the Monero address
        /// and to verify the private view key during a refund as the private view key may have been generated in a non standard way
        uint256 evmPublicViewKey;
        /// Monero private view key of the EVM side of the trade. Set during a call to refund.
        uint256 evmPrivateViewKey;
        /// Monero public spend key of the XMR side of the trade
        uint256 xmrPublicSpendKey;
        /// Monero private spend key of the XMR side of the trade. Set during a call to claim
        uint256 xmrPrivateSpendKey;
        /// Monero private view key of the XMR side of the trade. The EVM side of the trade doesn't need to share its private view key.
        uint256 xmrPrivateViewKey;
        /// Timestamp until which 'ready' can be called, after, taken offer is considered in the READY state
        uint256 t0;
        /// Timestamp until which 'claim' can be called. After, the EVM side can quit or the XMR side can resolve.
        uint256 t1;
    }

    struct Parameters {
        uint256 MINIMUM_OFFER;
        uint256 MAXIMUM_OFFER;
        uint256 DEPOSIT_RATIO;
        uint256 MAXIMUM_OFFER_BOOK_SIZE;
        uint256 T0_DELAY;
        uint256 T1_DELAY;
    }

    Parameters public parameters;

    /// Minimum T0 and T1 delays in seconds.
    uint256 constant MINIMUM_DELAY = 24 * 3_600;
    uint256 constant DEPOSIT_DENOMINATOR = 10000; // 10000 = 100%

    uint256 public liability;
    uint256 public nextOfferId = 1;
    mapping(uint256 => Offer) public offers;
    mapping(uint256 => bool) public usedPublicKeys;

    /// Mutex for non re-entrancy
    bool internal _mutex = false;
    modifier nonReentrant() {
        _nonReentrantBefore();
        _;
        _nonReentrantAfter();
    }

    function _nonReentrantBefore() internal {
        require(!_mutex, ErrorReentrancy());
        _mutex = true;
    }

    function _nonReentrantAfter() internal {
        _mutex = false;
    }

    event OfferEvent(uint256 offer_id, OfferType indexed kind, OfferState indexed state);

    constructor(Parameters memory _parameters, address _owner) payable {
        _initializeOwner(_owner);
        _setParameters(_parameters);
    }

    /// Receive function safeguard
    receive() external payable {
        require(0 == msg.value, ErrorUnableToAcceptPayment());
    }

    /// Fallback function safeguard
    fallback() external payable {
        require(0 == msg.value, ErrorUnableToAcceptPayment());
    }

    function _keySanity(uint256 pubKey) internal {
        require(!usedPublicKeys[pubKey], ErrorKeyAlreadyUsed());
        usedPublicKeys[pubKey] = true;
    }

    function openOffer(
        OfferType offerType,
        uint256 xmrAmount,
        address counterparty,
        uint256 spendingKey,
        uint256 viewingKey
    ) public payable returns (Offer memory offer) {
        require(
            0 == parameters.MAXIMUM_OFFER_BOOK_SIZE || nextOfferId <= parameters.MAXIMUM_OFFER_BOOK_SIZE,
            ErrorMaximumOfferBookSizeReached(nextOfferId)
        );

        _keySanity(spendingKey);
        _keySanity(viewingKey);

        if (offerType == OfferType.BUY) {
            offer.evmPublicSpendKey = spendingKey;
            offer.evmPublicViewKey = viewingKey;
        } else if (offerType == OfferType.SELL) {
            offer.xmrPublicSpendKey = spendingKey;
            offer.xmrPrivateViewKey = viewingKey;
        } else {
            revert ErrorInvalidOfferType();
        }

        uint256 evmAmount =
            offerType == OfferType.BUY ? msg.value : (msg.value * DEPOSIT_DENOMINATOR) / parameters.DEPOSIT_RATIO;
        uint256 deposit = offerType == OfferType.BUY
            ? ((msg.value * parameters.DEPOSIT_RATIO + DEPOSIT_DENOMINATOR - 1) / DEPOSIT_DENOMINATOR)
            : msg.value;

        require(evmAmount >= parameters.MINIMUM_OFFER && evmAmount <= parameters.MAXIMUM_OFFER, ErrorInvalidAmount());

        offer.kind = offerType;
        offer.id = nextOfferId++;
        offer.state = OfferState.OPEN;
        offer.lastupdate = block.timestamp;
        offer.owner = msg.sender;
        offer.counterparty = counterparty;
        offer.amount = evmAmount;
        offer.deposit = deposit;
        offer.xmrAmount = xmrAmount;

        liability += offer.kind == OfferType.BUY ? offer.amount : offer.deposit;

        offers[offer.id] = offer;

        emit OfferEvent(offer.id, offer.kind, offer.state);
    }

    /// Take an offer
    /// @param offerId Offer ID
    /// @param spendingKey (public)
    /// @param viewingKey (private (buy), public (sell))
    function take(uint256 offerId, uint256 spendingKey, uint256 viewingKey) public payable nonReentrant {
        Offer storage offer = offers[offerId];
        require(offer.state == OfferState.OPEN, ErrorOfferNotOpen());
        require(address(0) == offer.counterparty || offer.counterparty == msg.sender, ErrorNonMember());

        _keySanity(spendingKey);
        if (offer.kind == OfferType.BUY) {
            (uint256 x, uint256 y) = Ed25519.scalarMultBase(viewingKey);
            uint256 publicViewingKey = Ed25519.changeEndianness(Ed25519.compressPoint(x, y));
            _keySanity(publicViewingKey);
            offer.xmrPublicSpendKey = spendingKey;
            offer.xmrPrivateViewKey = viewingKey;
        } else {
            _keySanity(viewingKey);
            offer.evmPublicSpendKey = spendingKey;
            offer.evmPublicViewKey = viewingKey;
        }

        require(
            (offer.kind == OfferType.BUY && msg.value >= offer.deposit)
                || (offer.kind == OfferType.SELL && msg.value >= offer.amount),
            ErrorInvalidOfferAmount()
        );
        liability += offer.kind == OfferType.BUY ? offer.deposit : offer.amount;

        offer.state = OfferState.TAKEN;
        offer.counterparty = msg.sender;
        offer.blockTaken = block.number;
        offer.t0 = block.timestamp + parameters.T0_DELAY;
        offer.t1 = offer.t0 + parameters.T1_DELAY;
        offer.lastupdate = block.timestamp;

        emit OfferEvent(offer.id, offer.kind, offer.state);
    }

    /// Cancel an offer
    /// @param offerId Offer ID
    /// The deposit or amount will be returned to the caller
    function cancel(uint256 offerId) public nonReentrant {
        Offer storage offer = offers[offerId];
        require(offer.state == OfferState.OPEN, ErrorOfferNotOpen());
        require(offer.owner == msg.sender, ErrorNonMember());

        uint256 amount = offer.kind == OfferType.BUY ? offer.amount : offer.kind == OfferType.SELL ? offer.deposit : 0;
        require(amount > 0, ErrorInvalidOfferAmount());

        offer.state = OfferState.CANCELLED;
        offer.lastupdate = block.timestamp;

        liability -= amount;
        (bool res,) = payable(msg.sender).call{value: amount}("");
        require(res, ErrorUnableToRefund());

        emit OfferEvent(offer.id, offer.kind, offer.state);
    }

    /// Quit an offer
    /// Reveals private spending key and refunds both escrows
    /// @param offerId Offer ID
    /// @param spendingKey private spending key
    /// @param viewingKey private viewing key (0 for xmr-side)
    function quit(uint256 offerId, uint256 spendingKey, uint256 viewingKey) public nonReentrant {
        Offer storage offer = offers[offerId];

        if (
            (offer.kind == OfferType.BUY && msg.sender == offer.counterparty)
                || (offer.kind == OfferType.SELL && msg.sender == offer.owner)
        ) {
            // xmr
            require(offer.state == OfferState.READY || offer.state == OfferState.TAKEN, ErrorOfferNotReadyOrTaken());
            require(block.timestamp > offer.t1, ErrorClaimUnavailable());

            (uint256 x, uint256 y) = Ed25519.scalarMultBase(spendingKey);
            require(
                offer.xmrPublicSpendKey == Ed25519.changeEndianness(Ed25519.compressPoint(x, y)),
                ErrorInvalidPrivateSpendKey()
            );
            offer.xmrPrivateSpendKey = spendingKey;
        } else if (
            (offer.kind == OfferType.BUY && msg.sender == offer.owner)
                || (offer.kind == OfferType.SELL && msg.sender == offer.counterparty)
        ) {
            // evm
            require(
                (offer.state == OfferState.TAKEN && (block.timestamp <= offer.t0 || block.timestamp > offer.t1))
                    || (offer.kind != OfferType.INVALID
                        && offer.state == OfferState.READY
                        && block.timestamp > offer.t1),
                ErrorInvalidOfferStateForQuit()
            );
            require(
                offer.kind != OfferType.SELL || block.number > offer.blockTaken, ErrorSellOfferCannotQuitInTakenBlock()
            );

            (uint256 x, uint256 y) = Ed25519.scalarMultBase(spendingKey);
            require(
                offer.evmPublicSpendKey == Ed25519.changeEndianness(Ed25519.compressPoint(x, y)),
                ErrorBuyOfferInvalidEVMPrivateSpendKey()
            );
            (x, y) = Ed25519.scalarMultBase(viewingKey);
            require(
                offer.evmPublicViewKey == Ed25519.changeEndianness(Ed25519.compressPoint(x, y)),
                ErrorInvalidEVMPrivateViewKey()
            );
            offer.evmPrivateSpendKey = spendingKey;
            offer.evmPrivateViewKey = viewingKey;
        } else {
            revert ErrorNonMember();
        }

        offer.state = OfferState.REFUNDED;
        offer.lastupdate = block.timestamp;

        uint256 amountR1 = offer.kind == OfferType.BUY ? offer.amount : offer.kind == OfferType.SELL ? offer.deposit : 0;
        uint256 amountR2 = offer.kind == OfferType.BUY ? offer.deposit : offer.kind == OfferType.SELL ? offer.amount : 0;
        liability -= amountR1 + amountR2;
        (bool res,) = payable(offer.owner).call{value: amountR1}("");
        require(res, ErrorUnableToRefund());
        (bool res2,) = payable(offer.counterparty).call{value: amountR2}("");
        require(res2, ErrorUnableToRefund());

        emit OfferEvent(offer.id, offer.kind, offer.state);
    }

    /// Ready an offer
    /// This function is called by the buyer once the XMR deposit has been validated
    function ready(uint256 offerId) public nonReentrant {
        Offer storage offer = offers[offerId];
        require(offer.state == OfferState.TAKEN, ErrorOfferNotTaken());
        require(block.timestamp <= offer.t0, ErrorOfferAfterT0());
        require(
            (offer.kind == OfferType.BUY && msg.sender == offer.owner)
                || (offer.kind == OfferType.SELL && msg.sender == offer.counterparty),
            ErrorNonMember()
        );
        offer.state = OfferState.READY;
        offer.lastupdate = block.timestamp;

        emit OfferEvent(offer.id, offer.kind, offer.state);
    }

    /// Claim an offer
    /// @param offerId Offer ID
    /// @param privateSpendKey XMR private spend key
    /// The claim function completes a swap by revealing the private spend key
    function claim(uint256 offerId, uint256 privateSpendKey) public nonReentrant {
        Offer storage offer = offers[offerId];
        require(offer.state == OfferState.READY || offer.state == OfferState.TAKEN, ErrorOfferNotReadyOrTaken());

        require(
            (offer.state == OfferState.TAKEN && block.timestamp > offer.t0 && block.timestamp <= offer.t1)
                || (offer.state == OfferState.READY && block.timestamp <= offer.t1),
            ErrorClaimUnavailable()
        );
        require(
            (offer.kind == OfferType.BUY && msg.sender == offer.counterparty)
                || (offer.kind == OfferType.SELL && msg.sender == offer.owner),
            ErrorNonMember()
        );

        (uint256 x, uint256 y) = Ed25519.scalarMultBase(privateSpendKey);
        require(
            offer.xmrPublicSpendKey == Ed25519.changeEndianness(Ed25519.compressPoint(x, y)),
            ErrorInvalidPrivateSpendKey()
        );
        offer.xmrPrivateSpendKey = privateSpendKey;

        offer.state = OfferState.CLAIMED;
        offer.lastupdate = block.timestamp;

        uint256 amount = offer.amount + offer.deposit;
        liability -= amount;
        (bool res,) = payable(msg.sender).call{value: amount}("");
        require(res, ErrorUnableToPayClaimer());

        emit OfferEvent(offer.id, offer.kind, offer.state);
    }

    function recover() public onlyOwner {
        (bool res,) = payable(msg.sender).call{value: address(this).balance - liability}("");
        require(res, ErrorUnableToRefund());
    }

    function listOffers(uint256 offset, uint256 count, bool reverse) public view returns (Offer[] memory) {
        Offer[] memory _offers = new Offer[](count);
        for (uint256 i = 0; i < count; i++) {
            uint256 index = reverse ? count - i : i;
            _offers[i] = offers[index + offset];
        }
        return _offers;
    }

    function _setParameters(Parameters memory _parameters) internal {
        parameters = _parameters;
        require(parameters.T0_DELAY >= MINIMUM_DELAY, ErrorParametersInvalid());
        require(parameters.T1_DELAY >= MINIMUM_DELAY, ErrorParametersInvalid());
        require(
            parameters.DEPOSIT_RATIO > 0 && parameters.DEPOSIT_RATIO <= DEPOSIT_DENOMINATOR, ErrorParametersInvalid()
        );
    }

    function setParameters(Parameters memory _parameters) public onlyOwner {
        _setParameters(_parameters);
    }
}


// ===== FILE: src/Ed25519.sol =====
// From https://github.com/javgh/ed25519-solidity/blob/master/contract/Ed25519.sol
//
// Copyright (c) 2019, Jan Vornberger - licensed under the MIT license
//
// converted to a library by hbs in 2025 to avoid having to deploy a specific Ed22519 contract

pragma solidity ^0.8.34;

// Using formulas from https://hyperelliptic.org/EFD/g1p/auto-twisted-projective.html
// and constants from https://tools.ietf.org/html/draft-josefsson-eddsa-ed25519-03

library Ed25519 {
    uint256 constant q = 2 ** 255 - 19;
    uint256 constant d = 37095705934669439343138083508754565189542113879843219016388785533085940283555;
    // = -(121665/121666)
    uint256 constant Bx = 15112221349535400772501151409588531511454012693041857206046113283949847762202;
    uint256 constant By = 46316835694926478169428394003475163141307993866256225615783033603165251855960;

    struct Point {
        uint256 x;
        uint256 y;
        uint256 z;
    }

    struct Scratchpad {
        uint256 a;
        uint256 b;
        uint256 c;
        uint256 d;
        uint256 e;
        uint256 f;
        uint256 g;
        uint256 h;
    }

    function inv(uint256 a) internal view returns (uint256 invA) {
        uint256 e = q - 2;
        uint256 m = q;

        // use bigModExp precompile
        assembly ("memory-safe") {
            let p := mload(0x40)
            // WARNING: this line was added and magically made it work
            mstore(0x40, add(p, 0xc0))
            // THIS IS THE END OF THAT LINE THAT WAS ADDED thank u

            mstore(p, 0x20)
            mstore(add(p, 0x20), 0x20)
            mstore(add(p, 0x40), 0x20)
            mstore(add(p, 0x60), a)
            mstore(add(p, 0x80), e)
            mstore(add(p, 0xa0), m)
            if iszero(staticcall(not(0), 0x05, p, 0xc0, p, 0x20)) {
                revert(0, 0)
            }
            invA := mload(p)
        }
    }

    function ecAdd(Point memory p1, Point memory p2) internal pure returns (Point memory p3) {
        Scratchpad memory tmp;

        tmp.a = mulmod(p1.z, p2.z, q);
        tmp.b = mulmod(tmp.a, tmp.a, q);
        tmp.c = mulmod(p1.x, p2.x, q);
        tmp.d = mulmod(p1.y, p2.y, q);
        tmp.e = mulmod(d, mulmod(tmp.c, tmp.d, q), q);
        tmp.f = addmod(tmp.b, q - tmp.e, q);
        tmp.g = addmod(tmp.b, tmp.e, q);
        p3.x = mulmod(
            mulmod(tmp.a, tmp.f, q),
            addmod(addmod(mulmod(addmod(p1.x, p1.y, q), addmod(p2.x, p2.y, q), q), q - tmp.c, q), q - tmp.d, q),
            q
        );
        p3.y = mulmod(mulmod(tmp.a, tmp.g, q), addmod(tmp.d, tmp.c, q), q);
        p3.z = mulmod(tmp.f, tmp.g, q);
    }

    function ecDouble(Point memory p1) internal pure returns (Point memory p2) {
        Scratchpad memory tmp;

        tmp.a = addmod(p1.x, p1.y, q);
        tmp.b = mulmod(tmp.a, tmp.a, q);
        tmp.c = mulmod(p1.x, p1.x, q);
        tmp.d = mulmod(p1.y, p1.y, q);
        tmp.e = q - tmp.c;
        tmp.f = addmod(tmp.e, tmp.d, q);
        tmp.h = mulmod(p1.z, p1.z, q);
        tmp.g = addmod(tmp.f, q - mulmod(2, tmp.h, q), q);
        p2.x = mulmod(addmod(addmod(tmp.b, q - tmp.c, q), q - tmp.d, q), tmp.g, q);
        p2.y = mulmod(tmp.f, addmod(tmp.e, q - tmp.d, q), q);
        p2.z = mulmod(tmp.f, tmp.g, q);
    }

    function scalarMultBase(uint256 s) internal view returns (uint256, uint256) {
        Point memory b;
        Point memory result;
        b.x = Bx;
        b.y = By;
        b.z = 1;
        result.x = 0;
        result.y = 1;
        result.z = 1;

        while (s > 0) {
            if (s & 1 == 1) result = ecAdd(result, b);
            s = s >> 1;
            b = ecDouble(b);
        }

        uint256 invZ = inv(result.z);
        result.x = mulmod(result.x, invZ, q);
        result.y = mulmod(result.y, invZ, q);

        return (result.x, result.y);
    }

    function changeEndianness(uint256 _bigEnd) internal pure returns (uint256) {
        uint256 shifted = 0;
        uint256 i = 32;
        while (i > 0) {
            shifted <<= 8;
            shifted |= _bigEnd & 0xff;
            _bigEnd >>= 8;
            i--;
        }
        return shifted;
    }

    function compressPoint(uint256 x, uint256 y) internal pure returns (uint256) {
        uint256 compressed = y | ((x & 1) << 255);
        // Return is in Big Endian order - need to change endianness to stick to Monero's convention of using Little Endian
        return compressed;
    }
}


// ===== FILE: src/Errors.sol =====
// SPDX-License-Identifier: MIT
//
// Copyright (c) 2025-2026  v1rtl
//

pragma solidity ^0.8.34;

import "./Enums.sol";

//
// Errors
//

error ErrorNonMember();

error ErrorInvalidPrivateSpendKey();

error ErrorClaimUnavailable();

error ErrorOfferNotReadyOrTaken();

error ErrorOfferAfterT0();

error ErrorUnableToRefund();

error ErrorInvalidEVMPrivateViewKey();

error ErrorSellOfferCannotQuitInTakenBlock();

error ErrorOfferNotTaken();

error ErrorInvalidOfferStateForQuit();

error ErrorInvalidOfferAmount();

error ErrorOfferNotOpen();

error ErrorParametersInvalid();

error ErrorInvalidAmount();

/// Error raised when an address attempts to create a BuyOffer while there exists a FundingRequest for that same address.
/// The rationale is that if the address opens a BuyOffer then it has a balance which would have allowed to open a SellOffer.
/// So we disallow creating a BuyOffer unless the FundingRequest is removed, either voluntarily or because the SellOffer which
/// it funded has completed.
error ErrorBuyOfferNoCreationWhenActiveFundingRequestExists();

/// This error is raised when a BuyOffer is created or updated and the specified amount is below the minimum amount
/// configured in the contract for buy offers (MINIMUM_BUY_OFFER)
/// @param minimum the minimum acceptable amount
error ErrorBuyOfferAmountBelowMinimum(uint256 minimum);

/// This error is raised when a BuyOffer is created or updated and the specified amount is above the maximum amount
/// configured in the contract for buy offers (MAXIMUM_BUY_OFFER)
/// @param maximum the maximum acceptable amount
error ErrorBuyOfferAmountAboveMaximum(uint256 maximum);

/// This error is raised when a BuyOffer is updated and the specified state is not compatible with updates
/// @param state the current state of the offer
error ErrorBuyOfferInvalidStateForUpdate(OfferState state);

/// Error raised when attempting to reference a non existent BuyOffer.
error ErrorBuyOfferUnknown();

/// Error raised when attempting to update a BuyOffer from an address which is neither the owner nor the optionally configured manager
error ErrorBuyOfferInvalidCallerForUpdate();

/// Error raised when an operation that can only be performed by the buy offer owner is called from another address
error ErrorBuyOfferNotOwner();

/// Error raised when decreasing the maximum amount of a Buy Offer and the delta amount could not be sent back to the offer owner
error ErrorBuyOfferUnableToSendAmountDelta();

error ErrorKeyAlreadyUsed();

/// Error raised when the price specified in a take call is below the offer's fixed price
/// @param price the offer's fixed price
/// @param minprice the lower acceptable price specified by the taker
error ErrorBuyOfferPriceTooLow(uint256 price, uint256 minprice);

/// Error raised when the amount of XMR specified in a take call is below the minimum amount the buy offer owner is willing to buy
/// @param amount the amount of XMR specified in the take call
/// @param minamount the minimum amount of XMR the buy offer owner is willing to acquire
error ErrorBuyOfferXMRAmountTooLow(uint256 amount, uint256 minamount);

/// Error raised when a take operation is funded by a FundingRequest but the amount specified for the take doesn't cover the funding fee
error ErrorBuyOfferAmountTooLowToCoverFundingFee();

/// Error raised when a take operation is attempted on an offer which is not in the OPEN state.
/// @param state the current state of the offer
error ErrorBuyOfferInvalidStateForTake(OfferState state);

/// Error raised when a ready operation is attempted on an offer which is not in the TAKEN state.
/// @param state the current state of the offer
error ErrorBuyOfferInvalidStateForReady(OfferState state);

/// Error raised when attempting to call ready on a Buy Offer after the t0 timestamp
error ErrorBuyOfferAfterT0();

/// Error raised when attempting to call claim on a Buy Offer after the t1 timestamp
error ErrorBuyOfferAfterT1();

/// Error raised when attempting to call refund on a Buy Offer in the ready state on or before the t1 timestamp
error ErrorBuyOfferNotAfterT1();

/// Error raised when attempting to call claimDeposit before t1 if the offer was not refunded
error ErrorBuyOfferNotAfterT1OrRefunded();

/// Error raised when attempting to call refund on a Buy Offer when the current timestamp is > t0 and <= t1
error ErrorBuyOfferBetweenT0AndT1();

/// Error raised when attempting to call claim on a Buy Offer when the current timestamp is not > t0 and <= t1
error ErrorBuyOfferNotBetweenT0AndT1();

/// Error raised when the address calling claim is not the taker of the offer
error ErrorBuyOfferNotTaker();

/// Error raised when attempting to call claim on an offer which is not in a state from which it can be claimed (TAKEN or READY)
/// @param state the current state of the offer
error ErrorBuyOfferInvalidStateForClaim(OfferState state);

/// Error raised when attempting to call claimDeposit on a buy offer not in the READY or TAKEN state
error ErrorBuyOfferInvalidStateForClaimDeposit();

/// Error raised when attempting to call refund on an offer which is not in a state compatible with a refund call (TAKEN or READY)
/// @param state the current state of the offer
error ErrorBuyOfferInvalidStateForRefund(OfferState state);

/// Error raised when calling claim with a private spend key whose associated public spend key is not that specified when taking the offer
error ErrorBuyOfferInvalidXMRPrivateSpendKey();

/// Error raised when calling refund with a private spend key whose associated public spend key is not that specified when creating the offer
error ErrorBuyOfferInvalidEVMPrivateSpendKey();

/// Error raised when calling refund with a private view key whose associated public veiew key is not that specified when creating the offer
error ErrorBuyOfferInvalidEVMPrivateViewKey();

/// Error raised when attempting to cancel a buy offer which is not in the OPEN state
/// @param state the current state of the offer
error ErrorBuyOfferInvalidStateForCancel(OfferState state);

/// Error raised when attempting to take a buy offer while an unused fundind request exists for the caller
error ErrorBuyOfferAvailableFundingRequest();

/// Error raised when deposit could not be sent back to the caller during a cancel or refund call
error ErrorBuyOfferUnableToRefund();

/// Error raised when attempting to reduce an offer's maxamount value while transfering value in the tx
error ErrorBuyOfferNoValueAllowedWhenReducingMaxamount();

/// This error occurs when attempting to create a buy offer when the offer book is already at the configured limit
/// @param size the configured offer book maximum size
error ErrorMaximumOfferBookSizeReached(uint256 size);

/// This error is raised when price is 0 and oracleRatio is also 0
error ErrorBuyOfferNoPriceDefined();

/// This error is raised when a taker attempts to take a BuyOffer without sending any value with the transaction and
/// there is no FundingRequest for the taker's address.
error ErrorBuyOfferNoFundingRequestFound();

/// Error raised when the account taking the offer is not the specified counterparty
error ErrorBuyOfferInvalidCounterparty();

/// Error raised when there was an error sending back to the buyer the difference between the settlement amount and its deposit
error ErrorBuyOfferUnableToPayBuyer();

/// Error raised when attempting to call claimDeposit on a buy offer whose taker was funded
error ErrorBuyOfferCannotClaimDepositOfFundedOffer();

/// Error raised when attempting to create or update a sell offer with an amount below the configured minimum (MINIMUM_SELL_OFFER)
/// @param minimum the current configured minimum sell offer amount
error ErrorSellOfferAmountBelowMinimum(uint256 minimum);

/// Error raised when attempting to create or update a sell offer with an amount above the configured maximum (MAXIMUM_SELL_OFFER)
/// That error is also raised when attempting to take a sell offer with an amont above the offer's maximum
/// @param maximum the current configured maxmimum sell offer amount
error ErrorSellOfferAmountAboveMaximum(uint256 maximum);

/// Error raised when the account attempting to update a sell offer is neither its owner nor its manager
error ErrorSellOfferInvalidCallerForUpdate();

/// Error raised when attempting to update a sell offer which is not in the OPEN state
/// @param state the current state of the offer
error ErrorSellOfferInvalidStateForUpdate(OfferState state);

/// Error raised when attempting to perform an operation only avaialble to the owner of a Sell Offer from an address which is not the owner
error ErrorSellOfferNotOwner();

/// Error raised when cancelSellOffer was called by an account which is not the funder of the request or if the
/// required delay (2 * (T0_DELAY + T1_DELAY) has not passed since the funding request was funded.
error ErrorSellOfferNotCancellableByCaller();

/// Error raised when attempting to refund a sell offer from an address which is not the counterparty which took the offer
error ErrorSellOfferNotCounterparty();

/// Error raised when attempting to update the deposit amount of a sell offer which was created using a funding request
error ErrorSellOfferImmutableDeposit();

/// Error raised when the specified offer id is not a sell offer
error ErrorSellOfferUnknown();

/// Error raised when creating or updating a sell offer without specifying a price
error ErrorSellOfferNoPriceDefined();

/// Error raised when the amount resulting from taking a sell offer is not sufficient to cover the fees promised
/// to the funder of the FundingRequest which was used to create the sell offer.
/// This can be raised during calls to createSellOffer, updateSellOffer and takeSellOffer
error ErrorSellOfferAmountTooLowToCoverFundingFee();

/// Error raised when creating a sell offer with a public spend key which has already been used.
/// This is a check to ensure private spend keys are not reused across offers as this could lead to stolen funds.
error ErrorSellOfferPublicSpendKeyAlreadyUsed();

/// Error raised when the price resulting from taking a sell offer is above the taker's specified maximum
/// @param price the resulting price
/// @param maxprice the upper price limit which was specified by the taker
error ErrorSellOfferPriceTooHigh(uint256 price, uint256 maxprice);

/// Error raised when the resulting amount of XMR being sold is below the minimum specified by the buyer
/// @param amount resulting amount of XMR being sold
/// @param minimum cpecified minimum amount the buyer wants to acquire
error ErrorSellOfferXMRAmountTooLow(uint256 amount, uint256 minimum);

/// Error raised when the XMR amount the taker is willing to buy is below the minimum set by the maker
/// @param amount amount of XMR the taker is agreeing to buy
/// @param minimum the minimum amount of XMR the maker is willing to sell
error ErrorSellOfferXMRAmountBelowOfferMinimum(uint256 amount, uint256 minimum);

/// Error raised when attempting to take an offer which is not in the OPEN state
/// @param state the current state of the offer
error ErrorSellOfferInvalidStateForTake(OfferState state);

/// Error raised when attempting to call ready on an offer which is not in the TAKEN state
/// @param state the current state of the offer
error ErrorSellOfferInvalidStateForReady(OfferState state);

/// Error raised when attempting to call ready on an offer after the t0 timestamp
error ErrorSellOfferAfterT0();

/// Error raised when the account attempting the call ready on an offer is not its taker
error ErrorSellOfferNotTaker();

/// Error raised when attempting to claim an offer after the t1 timestamp
error ErrorSellOfferAfterT1();

/// Error raised when attempting to refund an offer on or before timestamp t1
error ErrorSellOfferNotAfterT1();

/// This error is also raised when attempting to call claimDeposit before t1 or if the offer is not refunded
error ErrorSellOfferNotAfterT1OrRefunded();

/// Error raised when attempting to claim an offer on or before t0 or after t1
error ErrorSellOfferNotBetweenT0AndT1();

/// Error raised when attempting to refund an offer after t0 and on or before t1
error ErrorSellOfferBetweenT0AndT1();

/// Error raised when attempting to cancel an offer which is not in the OPEN state
/// @param state the current state of the offer
error ErrorSellOfferInvalidStateForCancel(OfferState state);

/// Error raised when attempting to claim an offer which is not in the TAKEN or READY state
/// @param state the current state of the offer
error ErrorSellOfferInvalidStateForClaim(OfferState state);

/// Error raised when attempting to call claimDeposit on a sell offer not in the READY or TAKEN state
error ErrorSellOfferInvalidStateForClaimDeposit();

/// Error raised when attempting to refund an offer which is neither in the TAKEN nor READY state
/// @param state the current state of the offer
error ErrorSellOfferInvalidStateForRefund(OfferState state);

/// Error raised when attempting to claim an offer with a private spend key which is not associated with the public spend key specified at offer creation time
error ErrorSellOfferInvalidXMRPrivateSpendKey();

/// Error raised when attempting to refund an offer with a private spend key which is not associated with the public spend key specified when taking the offer
error ErrorSellOfferInvalidEVMPrivateSpendKey();

/// Error raised when attempting to refund an offer with a private view key which is not associated with the public view key specified when taking the offer
error ErrorSellOfferInvalidEVMPrivateViewKey();

/// Error raised when attempting to create a sell offer with a non 0 deposit while having a currently unused funding request
error ErrorSellOfferAvailableFundingRequest();

/// Error raised during cancel or refund calls when a deposit cannot be sent back
error ErrorSellOfferUnableToRefund();

/// Error raised when attempting to claim an offer which has already been claimed
error ErrorSellOfferAlreadyClaimed();

/// Error raised when attempting to refund an offer which has already been refunded
error ErrorSellOfferAlreadyRefunded();

/// Error raised when taking an offer with a deposit above the required one and when the delta couldn't be sent back to the taker
error ErrorSellOfferUnableToSendAmountDelta();

/// Error raised when the account taking the offer is not the specified counterparty
error ErrorSellOfferInvalidCounterparty();

/// Error raised when there was an error sending back to the buyer the difference between the settlement amount and its deposit
error ErrorSellOfferUnableToPayBuyer();

/// Error raised when a call to refund is performed in the same block in which take was called.
/// This is a mechanism to avoid having EVM takers call take and immediately call refund which would
/// simply be a way of draining the offer book to annoy sellers.
error ErrorSellOfferCannotRefundInTakenBlock();

/// Error raised when attempting to call claimDeposit on a sell offer whose maker was funded
error ErrorSellOfferCannotClaimDepositOfFundedOffer();

/// Generic error raised when an offer is invalid (either non existent, or not associated with caller)
error ErrorInvalidOffer();

/// Generic error raised when an offer is not of an expected type, most likely because it doesn not exist and is therefore of type INVALID
error ErrorInvalidOfferType();

/// Error raised when the payment of the claimer was unsuccessful
error ErrorUnableToPayClaimer();

/// Error raised when an offer deposit could not be claimed
error ErrorUnableToClaimDeposit();

/// Error raised when the payment of the funder was unsuccessful during a call to claim
error ErrorUnableToRepayFunder();

/// Error raised when attempting to perform an operation which is only available to the contract's owner from an account which is not that owner
error ErrorNotOwner();

/// This error is raised when attempting to set T0 or T1 delay to a value lower than MINIMUM_DELAY
/// @param delay the specified delay
/// @param minimum the configured minimum
error ErrorDelayTooShort(uint256 delay, uint256 minimum);

/// Error thrown when receiving value > 0 in either receive or fallback.
error ErrorUnableToAcceptPayment();

/// Error raised when reentrancy is detected
error ErrorReentrancy();


// ===== FILE: src/Enums.sol =====
// SPDX-License-Identifier: MIT
//
// Copyright (c) 2025-2026  v1rtl
//

pragma solidity ^0.8.34;

enum OfferType {
    INVALID, // Used so the default value 0 is invalid
    BUY,
    SELL
}

enum OfferState {
    INVALID, // Used so the default value 0 is invalid
    OPEN, // Open offers are those still seeking a counterparty
    TAKEN, // Taken offers are those with both a buyer and a seller
    CANCELLED, // Cancelled offers are those no longer valid
    REFUNDED, // Refunded offers are those for which the buyer requested a refund
    READY, // Ready offers are those for which the Monero deposit was confirmed by the buyer
    CLAIMED // Claimed offers are those whose Monero seller has claimed the amount of EVM currency paid for its XMR
}


// ===== FILE: lib/solady/src/auth/Ownable.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

/// @notice Simple single owner authorization mixin.
/// @author Solady (https://github.com/vectorized/solady/blob/main/src/auth/Ownable.sol)
///
/// @dev Note:
/// This implementation does NOT auto-initialize the owner to `msg.sender`.
/// You MUST call the `_initializeOwner` in the constructor / initializer.
///
/// While the ownable portion follows
/// [EIP-173](https://eips.ethereum.org/EIPS/eip-173) for compatibility,
/// the nomenclature for the 2-step ownership handover may be unique to this codebase.
abstract contract Ownable {
    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                       CUSTOM ERRORS                        */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev The caller is not authorized to call the function.
    error Unauthorized();

    /// @dev The `newOwner` cannot be the zero address.
    error NewOwnerIsZeroAddress();

    /// @dev The `pendingOwner` does not have a valid handover request.
    error NoHandoverRequest();

    /// @dev Cannot double-initialize.
    error AlreadyInitialized();

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                           EVENTS                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev The ownership is transferred from `oldOwner` to `newOwner`.
    /// This event is intentionally kept the same as OpenZeppelin's Ownable to be
    /// compatible with indexers and [EIP-173](https://eips.ethereum.org/EIPS/eip-173),
    /// despite it not being as lightweight as a single argument event.
    event OwnershipTransferred(address indexed oldOwner, address indexed newOwner);

    /// @dev An ownership handover to `pendingOwner` has been requested.
    event OwnershipHandoverRequested(address indexed pendingOwner);

    /// @dev The ownership handover to `pendingOwner` has been canceled.
    event OwnershipHandoverCanceled(address indexed pendingOwner);

    /// @dev `keccak256(bytes("OwnershipTransferred(address,address)"))`.
    uint256 private constant _OWNERSHIP_TRANSFERRED_EVENT_SIGNATURE =
        0x8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e0;

    /// @dev `keccak256(bytes("OwnershipHandoverRequested(address)"))`.
    uint256 private constant _OWNERSHIP_HANDOVER_REQUESTED_EVENT_SIGNATURE =
        0xdbf36a107da19e49527a7176a1babf963b4b0ff8cde35ee35d6cd8f1f9ac7e1d;

    /// @dev `keccak256(bytes("OwnershipHandoverCanceled(address)"))`.
    uint256 private constant _OWNERSHIP_HANDOVER_CANCELED_EVENT_SIGNATURE =
        0xfa7b8eab7da67f412cc9575ed43464468f9bfbae89d1675917346ca6d8fe3c92;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                          STORAGE                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev The owner slot is given by:
    /// `bytes32(~uint256(uint32(bytes4(keccak256("_OWNER_SLOT_NOT")))))`.
    /// It is intentionally chosen to be a high value
    /// to avoid collision with lower slots.
    /// The choice of manual storage layout is to enable compatibility
    /// with both regular and upgradeable contracts.
    bytes32 internal constant _OWNER_SLOT =
        0xffffffffffffffffffffffffffffffffffffffffffffffffffffffff74873927;

    /// The ownership handover slot of `newOwner` is given by:
    /// ```
    ///     mstore(0x00, or(shl(96, user), _HANDOVER_SLOT_SEED))
    ///     let handoverSlot := keccak256(0x00, 0x20)
    /// ```
    /// It stores the expiry timestamp of the two-step ownership handover.
    uint256 private constant _HANDOVER_SLOT_SEED = 0x389a75e1;

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                     INTERNAL FUNCTIONS                     */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Override to return true to make `_initializeOwner` prevent double-initialization.
    function _guardInitializeOwner() internal pure virtual returns (bool guard) {}

    /// @dev Initializes the owner directly without authorization guard.
    /// This function must be called upon initialization,
    /// regardless of whether the contract is upgradeable or not.
    /// This is to enable generalization to both regular and upgradeable contracts,
    /// and to save gas in case the initial owner is not the caller.
    /// For performance reasons, this function will not check if there
    /// is an existing owner.
    function _initializeOwner(address newOwner) internal virtual {
        if (_guardInitializeOwner()) {
            /// @solidity memory-safe-assembly
            assembly {
                let ownerSlot := _OWNER_SLOT
                if sload(ownerSlot) {
                    mstore(0x00, 0x0dc149f0) // `AlreadyInitialized()`.
                    revert(0x1c, 0x04)
                }
                // Clean the upper 96 bits.
                newOwner := shr(96, shl(96, newOwner))
                // Store the new value.
                sstore(ownerSlot, or(newOwner, shl(255, iszero(newOwner))))
                // Emit the {OwnershipTransferred} event.
                log3(0, 0, _OWNERSHIP_TRANSFERRED_EVENT_SIGNATURE, 0, newOwner)
            }
        } else {
            /// @solidity memory-safe-assembly
            assembly {
                // Clean the upper 96 bits.
                newOwner := shr(96, shl(96, newOwner))
                // Store the new value.
                sstore(_OWNER_SLOT, newOwner)
                // Emit the {OwnershipTransferred} event.
                log3(0, 0, _OWNERSHIP_TRANSFERRED_EVENT_SIGNATURE, 0, newOwner)
            }
        }
    }

    /// @dev Sets the owner directly without authorization guard.
    function _setOwner(address newOwner) internal virtual {
        if (_guardInitializeOwner()) {
            /// @solidity memory-safe-assembly
            assembly {
                let ownerSlot := _OWNER_SLOT
                // Clean the upper 96 bits.
                newOwner := shr(96, shl(96, newOwner))
                // Emit the {OwnershipTransferred} event.
                log3(0, 0, _OWNERSHIP_TRANSFERRED_EVENT_SIGNATURE, sload(ownerSlot), newOwner)
                // Store the new value.
                sstore(ownerSlot, or(newOwner, shl(255, iszero(newOwner))))
            }
        } else {
            /// @solidity memory-safe-assembly
            assembly {
                let ownerSlot := _OWNER_SLOT
                // Clean the upper 96 bits.
                newOwner := shr(96, shl(96, newOwner))
                // Emit the {OwnershipTransferred} event.
                log3(0, 0, _OWNERSHIP_TRANSFERRED_EVENT_SIGNATURE, sload(ownerSlot), newOwner)
                // Store the new value.
                sstore(ownerSlot, newOwner)
            }
        }
    }

    /// @dev Throws if the sender is not the owner.
    function _checkOwner() internal view virtual {
        /// @solidity memory-safe-assembly
        assembly {
            // If the caller is not the stored owner, revert.
            if iszero(eq(caller(), sload(_OWNER_SLOT))) {
                mstore(0x00, 0x82b42900) // `Unauthorized()`.
                revert(0x1c, 0x04)
            }
        }
    }

    /// @dev Returns how long a two-step ownership handover is valid for in seconds.
    /// Override to return a different value if needed.
    /// Made internal to conserve bytecode. Wrap it in a public function if needed.
    function _ownershipHandoverValidFor() internal view virtual returns (uint64) {
        return 48 * 3600;
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                  PUBLIC UPDATE FUNCTIONS                   */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Allows the owner to transfer the ownership to `newOwner`.
    function transferOwnership(address newOwner) public payable virtual onlyOwner {
        /// @solidity memory-safe-assembly
        assembly {
            if iszero(shl(96, newOwner)) {
                mstore(0x00, 0x7448fbae) // `NewOwnerIsZeroAddress()`.
                revert(0x1c, 0x04)
            }
        }
        _setOwner(newOwner);
    }

    /// @dev Allows the owner to renounce their ownership.
    function renounceOwnership() public payable virtual onlyOwner {
        _setOwner(address(0));
    }

    /// @dev Request a two-step ownership handover to the caller.
    /// The request will automatically expire in 48 hours (172800 seconds) by default.
    function requestOwnershipHandover() public payable virtual {
        unchecked {
            uint256 expires = block.timestamp + _ownershipHandoverValidFor();
            /// @solidity memory-safe-assembly
            assembly {
                // Compute and set the handover slot to `expires`.
                mstore(0x0c, _HANDOVER_SLOT_SEED)
                mstore(0x00, caller())
                sstore(keccak256(0x0c, 0x20), expires)
                // Emit the {OwnershipHandoverRequested} event.
                log2(0, 0, _OWNERSHIP_HANDOVER_REQUESTED_EVENT_SIGNATURE, caller())
            }
        }
    }

    /// @dev Cancels the two-step ownership handover to the caller, if any.
    function cancelOwnershipHandover() public payable virtual {
        /// @solidity memory-safe-assembly
        assembly {
            // Compute and set the handover slot to 0.
            mstore(0x0c, _HANDOVER_SLOT_SEED)
            mstore(0x00, caller())
            sstore(keccak256(0x0c, 0x20), 0)
            // Emit the {OwnershipHandoverCanceled} event.
            log2(0, 0, _OWNERSHIP_HANDOVER_CANCELED_EVENT_SIGNATURE, caller())
        }
    }

    /// @dev Allows the owner to complete the two-step ownership handover to `pendingOwner`.
    /// Reverts if there is no existing ownership handover requested by `pendingOwner`.
    function completeOwnershipHandover(address pendingOwner) public payable virtual onlyOwner {
        /// @solidity memory-safe-assembly
        assembly {
            // Compute and set the handover slot to 0.
            mstore(0x0c, _HANDOVER_SLOT_SEED)
            mstore(0x00, pendingOwner)
            let handoverSlot := keccak256(0x0c, 0x20)
            // If the handover does not exist, or has expired.
            if gt(timestamp(), sload(handoverSlot)) {
                mstore(0x00, 0x6f5e8818) // `NoHandoverRequest()`.
                revert(0x1c, 0x04)
            }
            // Set the handover slot to 0.
            sstore(handoverSlot, 0)
        }
        _setOwner(pendingOwner);
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                   PUBLIC READ FUNCTIONS                    */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Returns the owner of the contract.
    function owner() public view virtual returns (address result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := sload(_OWNER_SLOT)
        }
    }

    /// @dev Returns the expiry timestamp for the two-step ownership handover to `pendingOwner`.
    function ownershipHandoverExpiresAt(address pendingOwner)
        public
        view
        virtual
        returns (uint256 result)
    {
        /// @solidity memory-safe-assembly
        assembly {
            // Compute the handover slot.
            mstore(0x0c, _HANDOVER_SLOT_SEED)
            mstore(0x00, pendingOwner)
            // Load the handover slot.
            result := sload(keccak256(0x0c, 0x20))
        }
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                         MODIFIERS                          */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @dev Marks a function as only callable by the owner.
    modifier onlyOwner() virtual {
        _checkOwner();
        _;
    }
}
