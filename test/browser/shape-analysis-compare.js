export function compareContourAnalysis(actual, reference, known) {
  if (actual.length !== reference.length) throw new Error("Shape case count differs");
  const differences = new Map(known.map((entry) => [entry.actual.name, entry]));
  let numericCases = 0;
  let errorCases = 0;
  const observed = [];
  for (let index = 0; index < actual.length; index++) {
    const value = actual[index],
      expected = reference[index];
    if (value.name !== expected.name) throw new Error("Shape case ordering differs");
    const recorded = differences.get(value.name);
    if (JSON.stringify(value) === JSON.stringify(expected)) {
      if (recorded)
        throw new Error(`${value.name}: recorded difference resolved; update the ledger`);
      if (value.error) errorCases++;
      else numericCases++;
    } else {
      if (
        !recorded ||
        JSON.stringify(value) !== JSON.stringify(recorded.actual) ||
        JSON.stringify(expected) !== JSON.stringify(recorded.expected)
      ) {
        throw new Error(
          `${value.name}: unrecorded difference\n${JSON.stringify({ actual: value, expected })}`,
        );
      }
      observed.push({ name: value.name, reason: recorded.reason });
      differences.delete(value.name);
    }
  }
  if (differences.size) throw new Error("Recorded difference cases were not exercised");
  return { cases: actual.length, numericCases, errorCases, knownDifferences: observed };
}
