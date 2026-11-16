// Copyright 2026 The uTLS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"math/rand"
	"testing"
)

// shuffleTestSeed is the fixed seed for the distribution regression tests, so
// that every run exercises the exact same sequence of shuffles.
const shuffleTestSeed int64 = 0x5eed_0094_0000_0001

// newShuffleTestSpec builds the canonical layout of a modern Chrome
// ClientHello: pinned extensions (GREASE at the head, GREASE and
// pre_shared_key at the tail) interleaved with movable extensions. The
// concrete extension values are irrelevant here; only their types and
// positions matter to the shuffle.
func newShuffleTestSpec() []TLSExtension {
	return []TLSExtension{
		&UtlsGREASEExtension{}, // slot 0: pinned
		&SNIExtension{},        // slot 1: movable (id 0)
		&ExtendedMasterSecretExtension{},
		&RenegotiationInfoExtension{},
		&SupportedCurvesExtension{},
		&SupportedPointsExtension{},
		&SessionTicketExtension{},
		&ALPNExtension{},
		&StatusRequestExtension{},
		&SignatureAlgorithmsExtension{},
		&SCTExtension{},
		&KeyShareExtension{},
		&PSKKeyExchangeModesExtension{},
		&SupportedVersionsExtension{},
		&UtlsCompressCertExtension{},
		&ApplicationSettingsExtension{}, // slot 15: movable (id 14, last movable)
		&UtlsGREASEExtension{},          // slot 16: pinned
		&UtlsPreSharedKeyExtension{},    // slot 17: pinned (PreSharedKeyExtension)
	}
}

// cloneShuffleTestSpec returns a fresh slice referencing the same extension
// pointers as base. The shuffle only permutes pointers, so sharing pointers
// lets the tests identify each extension by pointer across iterations.
func cloneShuffleTestSpec(base []TLSExtension) []TLSExtension {
	return append([]TLSExtension(nil), base...)
}

// movableShuffleIDs maps every movable slot in the spec to a stable label
// derived from its original position. Pinned slots are absent from the map.
func movableShuffleIDs(exts []TLSExtension) map[int]int {
	ids := make(map[int]int)
	movableID := 0
	for idx := range exts {
		if isChromeShufflePinned(idx, exts) {
			continue
		}
		ids[idx] = movableID
		movableID++
	}
	return ids
}

// pinnedShuffleSlots returns the absolute slots occupied by pinned extensions.
func pinnedShuffleSlots(exts []TLSExtension) []int {
	var slots []int
	for idx := range exts {
		if isChromeShufflePinned(idx, exts) {
			slots = append(slots, idx)
		}
	}
	return slots
}

// originSlotOf finds the slot the extension occupied in the original spec,
// identified by pointer equality: the shuffle only permutes the existing
// pointers, it never allocates replacement extensions.
func originSlotOf(ext TLSExtension, base []TLSExtension) int {
	for i, b := range base {
		if b == ext {
			return i
		}
	}
	return -1
}

// TestShuffleChromeTLSExtensionsUniformDistribution is the distribution
// regression test for the Chrome extension shuffle.
//
// Uniformity requirement: every movable extension must land in every movable
// slot with the same probability 1/n, where n is the number of movable
// extensions. The previous implementation ran Fisher-Yates over the whole
// slice and silently skipped swaps touching a pinned slot; a skipped swap
// leaves an element stranded in place with above-uniform probability, and the
// strength of the bias depends on the slot's distance from the pinned slots,
// producing a position-dependent frequency fingerprint observable over
// thousands of handshakes even though every single handshake remains
// byte-identical in content to a real Chrome one.
//
// Test design (no network needed, safe under -short):
//
//   - Fixed seed: a single math/rand source seeded with shuffleTestSeed drives
//     every shuffle through shuffleChromeTLSExtensionsWithRand, so the run is
//     fully deterministic and reproducible.
//   - Sample size: N = 200000 shuffles over n = 15 movable extensions (the
//     movable set of a modern Chrome ClientHello after excluding its 3 pinned
//     slots). Under the uniform null hypothesis each (extension, movable
//     slot) cell is Binomial(N, 1/n) with expected count N/n ≈ 13333 and
//     standard deviation sqrt(N*(1/n)*(1-1/n)) ≈ 111.6.
//   - Tolerance band: ±5 relative percent around the expected frequency
//     (counts within [12667, 14000]), roughly ±6 standard deviations. A
//     correctly uniform shuffle breaches this band only with astronomically
//     small probability, while the skipped-swap bias shifts stationary counts
//     by far more than 5% for a layout with 15 movable elements and 3 pinned
//     slots, so the band cleanly separates the two implementations.
//   - Every shuffle is additionally checked to remain a bijection: each
//     movable slot holds exactly one of the movable extensions afterwards.
func TestShuffleChromeTLSExtensionsUniformDistribution(t *testing.T) {
	const (
		samples   = 200000
		tolerance = 0.05 // ±5% relative band around the uniform frequency
	)

	base := newShuffleTestSpec()
	slotToID := movableShuffleIDs(base)
	n := len(slotToID)
	if n == 0 {
		t.Fatal("test spec has no movable extensions")
	}

	// count[id][movable column] = number of observed landings.
	counts := make([][]int, n)
	for i := range counts {
		counts[i] = make([]int, n)
	}

	rng := rand.New(rand.NewSource(shuffleTestSeed))
	for iter := 0; iter < samples; iter++ {
		exts := cloneShuffleTestSpec(base)
		shuffleChromeTLSExtensionsWithRand(exts, rng)

		seen := make(map[int]bool, n) // movable id -> seen in this shuffle
		movableCol := 0
		for slot, ext := range exts {
			if isChromeShufflePinned(slot, exts) {
				continue
			}
			origin := originSlotOf(ext, base)
			id, ok := slotToID[origin]
			if !ok {
				t.Fatalf("iteration %d: extension at slot %d is not one of the movable extensions (origin=%d)",
					iter, slot, origin)
			}
			if seen[id] {
				t.Fatalf("iteration %d: movable extension id %d appeared twice after shuffle (not a permutation)", iter, id)
			}
			seen[id] = true
			counts[id][movableCol]++
			movableCol++
		}
		if len(seen) != n {
			t.Fatalf("iteration %d: shuffle dropped/duplicated extensions: got %d distinct movable, want %d",
				iter, len(seen), n)
		}
	}

	expected := float64(samples) / float64(n)
	lo := expected * (1 - tolerance)
	hi := expected * (1 + tolerance)
	for id := 0; id < n; id++ {
		for col := 0; col < n; col++ {
			got := float64(counts[id][col])
			if got < lo || got > hi {
				t.Errorf("non-uniform placement: movable extension id %d at movable slot %d "+
					"observed %d times (freq %.4f), want uniform %.0f within ±%.0f%% band [%.0f, %.0f]",
					id, col, counts[id][col], got/float64(samples), expected,
					tolerance*100, lo, hi)
			}
		}
	}

	// Column marginal check: each movable slot receives exactly one extension
	// per shuffle, so its total must equal the sample count.
	for col := 0; col < n; col++ {
		total := 0
		for id := 0; id < n; id++ {
			total += counts[id][col]
		}
		if total != samples {
			t.Errorf("movable slot %d received %d extensions, want %d", col, total, samples)
		}
	}
	t.Logf("uniformity: %d samples, %d movable extensions, expected cell count %.1f, band [%.1f, %.1f]",
		samples, n, expected, lo, hi)
}

// TestShuffleChromeTLSExtensionsPinnedInvariance asserts that pinned
// extensions (GREASE, padding, pre_shared_key) never move and never change
// identity during a shuffle, while the multiset of extension pointers over the
// whole slice is preserved.
func TestShuffleChromeTLSExtensionsPinnedInvariance(t *testing.T) {
	base := newShuffleTestSpec()
	pinnedSlots := pinnedShuffleSlots(base)
	if len(pinnedSlots) != 3 {
		t.Fatalf("test spec changed: got %d pinned slots, want 3", len(pinnedSlots))
	}

	rng := rand.New(rand.NewSource(shuffleTestSeed))
	for iter := 0; iter < 1000; iter++ {
		exts := cloneShuffleTestSpec(base)
		shuffleChromeTLSExtensionsWithRand(exts, rng)

		if len(exts) != len(base) {
			t.Fatalf("iteration %d: shuffle changed extension count: got %d, want %d",
				iter, len(exts), len(base))
		}

		// Pinned slots must hold the exact same extension pointer as before
		// and remain semantically pinned.
		for _, slot := range pinnedSlots {
			if exts[slot] != base[slot] {
				t.Fatalf("iteration %d: pinned extension at slot %d moved or was replaced", iter, slot)
			}
			if !isChromeShufflePinned(slot, exts) {
				t.Fatalf("iteration %d: slot %d lost its pinned semantics", iter, slot)
			}
		}

		// The multiset of extension pointers over the whole slice is preserved.
		remaining := make(map[TLSExtension]int)
		for _, ext := range base {
			remaining[ext]++
		}
		for _, ext := range exts {
			if remaining[ext] == 0 {
				t.Fatalf("iteration %d: shuffle introduced an unknown extension", iter)
			}
			remaining[ext]--
		}
		for ext, count := range remaining {
			if count != 0 {
				t.Fatalf("iteration %d: extension %v lost during shuffle (remaining=%d)", iter, ext, count)
			}
		}
	}
}

// TestShuffleChromeTLSExtensionsEdgeCases covers degenerate layouts: all
// pinned, a single movable element, and an empty slice. None must panic and
// pinned semantics must hold.
func TestShuffleChromeTLSExtensionsEdgeCases(t *testing.T) {
	rng := rand.New(rand.NewSource(1))

	t.Run("all_pinned", func(t *testing.T) {
		exts := []TLSExtension{
			&UtlsGREASEExtension{},
			&UtlsPaddingExtension{},
			&UtlsPreSharedKeyExtension{},
		}
		want := append([]TLSExtension(nil), exts...)
		shuffleChromeTLSExtensionsWithRand(exts, rng)
		for i := range exts {
			if exts[i] != want[i] {
				t.Fatalf("pinned extension at slot %d moved", i)
			}
		}
	})

	t.Run("single_movable", func(t *testing.T) {
		exts := []TLSExtension{
			&UtlsGREASEExtension{},
			&SNIExtension{},
			&UtlsPreSharedKeyExtension{},
		}
		want := append([]TLSExtension(nil), exts...)
		shuffleChromeTLSExtensionsWithRand(exts, rng)
		for i := range exts {
			if exts[i] != want[i] {
				t.Fatalf("layout changed with only one movable extension at slot %d", i)
			}
		}
	})

	t.Run("empty", func(t *testing.T) {
		exts := []TLSExtension{}
		shuffleChromeTLSExtensionsWithRand(exts, rng)
		if len(exts) != 0 {
			t.Fatalf("empty slice gained elements: %d", len(exts))
		}
	})
}
